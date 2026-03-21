"""Knowledge, session, and mesh command handlers.

/knowledge, /session, /sessions, /mesh
"""

from __future__ import annotations


class KnowledgeMixin:
    """Mixin providing /knowledge, /session, /sessions, /mesh command handlers."""

    # -------------------------------------------------------------------
    # /knowledge
    # -------------------------------------------------------------------

    def _handle_knowledge(self, args: list[str]) -> str:
        if not self.knowledge_store:
            return "Knowledge store not available."

        query = " ".join(args) if args else ""

        try:
            claims = self.knowledge_store.list_claims(
                scope_id=self.project_id,
                status="accepted",
                limit=100,
            )

            if query:
                query_lower = query.lower()
                claims = [c for c in claims if query_lower in c.statement.lower()]

            if not claims:
                return f"No claims found{' matching: ' + query if query else ''}."

            lines = [f"Knowledge ({len(claims)} claims):"]
            for c in claims[:10]:
                key = c.canonical.claim_key if c.canonical else "?"
                stmt = c.statement[:120]
                lines.append(f"  [{key}] {stmt}")

            if len(claims) > 10:
                lines.append(f"  ... and {len(claims) - 10} more")

            return "\n".join(lines)
        except Exception as e:
            return f"Error querying knowledge: {e}"

    # -------------------------------------------------------------------
    # /session, /sessions
    # -------------------------------------------------------------------

    async def _handle_session(self, args: list[str] | None = None) -> str:
        if not args:
            if not self.session_id:
                return "No active session."
            lines = [f"Session: {self.session_id}"]
            # Show message count if shell has session manager
            if self.shell and hasattr(self.shell, "sessions") and self.shell.sessions:
                session = self.shell.sessions.get_session(self.session_id)
                if session:
                    lines.append(f"Messages: {len(session.messages)}")
            return "\n".join(lines)

        subcmd = args[0].lower()

        if subcmd == "export":
            return self._export_session()

        if subcmd == "reset":
            export_msg = self._export_session()
            # Reset Claude conversation ID to start fresh
            if self.shell and hasattr(self.shell, "_claude_conversation_id"):
                self.shell._claude_conversation_id = f"shell-{self.session_id}-{__import__('time').time():.0f}"
            if self.shell and hasattr(self.shell, "_claude_adapter") and self.shell._claude_adapter:
                self.shell._claude_adapter._sessions.clear()
            return f"{export_msg}\nSession reset. Claude brain will start a fresh conversation."

        if subcmd == "capture":
            return self._capture_sdk_session(args[1] if len(args) > 1 else None)

        if subcmd in ("sdk-list", "sdk"):
            return self._list_sdk_sessions()

        return "Usage: /session [export|reset|capture [session_id]|sdk-list]"

    def _list_sdk_sessions(self) -> str:
        """List recent Claude SDK session files."""
        from ..session_capture import SessionCapture
        capture = SessionCapture()
        sessions = capture.list_sessions(limit=10, min_size=5000)
        if not sessions:
            return "No SDK session files found."
        lines = ["SDK Sessions (recent, >5KB):"]
        for s in sessions:
            lines.append(f"  {s['session_id'][:12]}... {s['size_kb']:>8.1f} KB  {s['modified'][:16]}")
        lines.append(f"\nUse /session capture <session_id> to export & ingest.")
        return "\n".join(lines)

    def _capture_sdk_session(self, sdk_session_id: str | None) -> str:
        """Capture an SDK session: export markdown + ingest claims."""
        from ..session_capture import SessionCapture, ingest_session_to_knowledge

        capture = SessionCapture()

        # If no session_id given, try to find the current shell's SDK session
        if not sdk_session_id:
            if (self.shell
                    and hasattr(self.shell, "_claude_adapter")
                    and self.shell._claude_adapter):
                sdk_session_id = self.shell._claude_adapter.get_session_id("shell-default")
            if not sdk_session_id:
                # Use the most recent large session
                sessions = capture.list_sessions(limit=1, min_size=5000)
                if sessions:
                    sdk_session_id = sessions[0]["session_id"]

        if not sdk_session_id:
            return "No SDK session found. Provide a session_id: /session capture <id>"

        lines = []

        # Export to markdown
        try:
            export_path = capture.export_markdown(sdk_session_id)
            stats = capture.get_session_stats(sdk_session_id)
            lines.append(f"Exported: {export_path}")
            lines.append(f"  Messages: {stats['user_messages']} user + {stats['assistant_messages']} assistant")
            lines.append(f"  Tools used: {', '.join(stats['unique_tools'][:10])}")
            lines.append(f"  Compactions: {stats['compactions']}")
            lines.append(f"  Est. tokens: {stats['estimated_tokens']:,}")
        except Exception as e:
            lines.append(f"Export failed: {e}")

        # Ingest into knowledge.db
        if self.knowledge_store:
            try:
                claim_ids = ingest_session_to_knowledge(
                    sdk_session_id, self.knowledge_store,
                )
                lines.append(f"  Ingested: {len(claim_ids)} claims into knowledge.db")
            except Exception as e:
                lines.append(f"  Ingestion failed: {e}")
        else:
            lines.append("  Ingestion skipped (no knowledge store)")

        return "\n".join(lines)

    def _export_session(self) -> str:
        """Export current session transcript to a markdown file."""
        if not self.session_id or not self.shell:
            return "No active session to export."

        sm = getattr(self.shell, "sessions", None)
        if not sm:
            return "Session manager not available."

        session = sm.get_session(self.session_id)
        if not session or not session.messages:
            return "Session has no messages to export."

        # Build markdown
        lines = [
            f"# Session Export: {self.session_id}",
            f"Exported: {__import__('datetime').datetime.now().isoformat()}",
            f"Messages: {len(session.messages)}",
            "",
            "---",
            "",
        ]
        for msg in session.messages:
            role = msg.role.upper()
            lines.append(f"### {role}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Write to data_dir/sessions/exports/
        export_dir = sm.sessions_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = __import__("time").strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_{ts}.md"
        export_path = export_dir / filename
        export_path.write_text("\n".join(lines), encoding="utf-8")

        return f"Session exported to {export_path}"

    def _handle_sessions(self) -> str:
        """List available sessions (reads session files from disk)."""
        lines = ["Sessions:"]
        # SessionManager doesn't have list_sessions(), so this is minimal
        lines.append(f"  Current: {self.session_id or 'none'}")
        lines.append("  Use --session <id> when starting shell to resume.")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # /mesh
    # -------------------------------------------------------------------

    def _handle_mesh(self) -> str:
        if not self.knowledge_store:
            return "Knowledge store not available (needed for peer discovery)."

        try:
            peers = self.knowledge_store.get_presence_peers()
            if not peers:
                return "No mesh peers detected."

            lines = [f"Mesh Peers ({len(peers)}):"]
            for p in peers:
                node = getattr(p, "node_id", str(p))
                status = getattr(p, "status", "unknown")
                lines.append(f"  {node} [{status}]")
            return "\n".join(lines)
        except Exception as e:
            return f"Error querying mesh: {e}"
