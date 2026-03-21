"""Process, service, and pipeline command handlers.

/spawn, /ps, /output, /kill, /services, /pipeline
"""

from __future__ import annotations


class ProcessMixin:
    """Mixin providing process/service/pipeline command handlers."""

    # -------------------------------------------------------------------
    # /spawn, /ps, /output, /kill
    # -------------------------------------------------------------------

    async def _handle_spawn(self, args: list[str]) -> str:
        if not args:
            return "Usage: /spawn <command>"
        if not self.process_manager:
            return "Process manager not available. Use `multihead shell` for process management."
        command = " ".join(args)
        try:
            proc = await self.process_manager.spawn(command)
            return f"Started PID {proc.pid}: {command}"
        except Exception as e:
            return f"Failed to spawn: {e}"

    def _handle_ps(self) -> str:
        if not self.process_manager:
            return "Process manager not available."
        procs = self.process_manager.list_processes()
        if not procs:
            return "No tracked processes."
        lines = ["Processes:"]
        for p in procs:
            info = p.to_dict()
            lines.append(
                f"  PID {info['pid']}: {info['command'][:60]} "
                f"[{info['status']}] {info['elapsed_seconds']}s"
            )
        return "\n".join(lines)

    def _handle_output(self, args: list[str]) -> str:
        if not args:
            return "Usage: /output <pid> [lines]"
        if not self.process_manager:
            return "Process manager not available."
        try:
            pid = int(args[0])
        except ValueError:
            return "PID must be an integer."
        lines = int(args[1]) if len(args) > 1 else 20
        return self.process_manager.output(pid, lines)

    def _handle_kill(self, args: list[str]) -> str:
        if not args:
            return "Usage: /kill <pid>"
        if not self.process_manager:
            return "Process manager not available."
        try:
            pid = int(args[0])
        except ValueError:
            return "PID must be an integer."
        return self.process_manager.kill(pid)

    # -------------------------------------------------------------------
    # /pipeline
    # -------------------------------------------------------------------

    def _handle_pipeline(self, args: list[str]) -> str:
        """Handle /pipeline — show status, toggle, or set config."""
        if not args:
            # Show pipeline status
            lines = []
            pipeline_cfg = getattr(self.config, "pipeline", None)
            if pipeline_cfg is None:
                return "Pipeline config not available."

            lines.append(f"Pipeline: {'ON' if pipeline_cfg.enabled else 'OFF'}")
            lines.append(f"  knowledge_rag: {pipeline_cfg.knowledge_rag}")
            lines.append(f"  auto_decompose: {pipeline_cfg.auto_decompose}")
            lines.append(f"  auto_record: {pipeline_cfg.auto_record}")
            lines.append(f"  vlm_auto_route: {pipeline_cfg.vlm_auto_route}")
            lines.append(f"  decompose_threshold: {pipeline_cfg.decompose_threshold}")
            head_label = pipeline_cfg.decompose_head or "(auto-select)"
            lines.append(f"  decompose_head: {head_label}")

            # Show stats if shell has pipeline
            if self.shell and hasattr(self.shell, "pipeline") and self.shell.pipeline:
                lines.append("")
                lines.append(self.shell.pipeline.stats_summary())

            return "\n".join(lines)

        subcmd = args[0].lower()

        if subcmd in ("on", "off"):
            self.config.pipeline.enabled = (subcmd == "on")
            self._save_config()
            return f"Pipeline {'enabled' if self.config.pipeline.enabled else 'disabled'}."

        if subcmd == "set" and len(args) >= 3:
            key = args[1]
            value = args[2]
            try:
                result = self.config.set_value(f"pipeline.{key}", value)
                self._save_config()
                return result
            except ValueError as e:
                return str(e)

        return "Usage: /pipeline [on|off] | /pipeline set <key> <value>"

    # -------------------------------------------------------------------
    # /services
    # -------------------------------------------------------------------

    async def _handle_services(self, args: list[str]) -> str:
        """Handle /services — list, start, stop, enable, disable."""
        if not self.service_manager:
            return "Service manager not available."

        if not args or args[0] == "list":
            return self._format_services_list()

        subcmd = args[0].lower()

        if subcmd == "start" and len(args) >= 2:
            name = args[1]
            return await self.service_manager.start(name)

        if subcmd == "stop" and len(args) >= 2:
            name = args[1]
            return await self.service_manager.stop(name)

        if subcmd == "enable" and len(args) >= 2:
            name = args[1]
            config_key = name.replace("-", "_")
            try:
                result = self.config.set_value(f"services.{config_key}", "true")
                self._save_config()
                return f"Auto-start enabled for '{name}'. {result}"
            except ValueError as e:
                return str(e)

        if subcmd == "disable" and len(args) >= 2:
            name = args[1]
            config_key = name.replace("-", "_")
            try:
                result = self.config.set_value(f"services.{config_key}", "false")
                self._save_config()
                return f"Auto-start disabled for '{name}'. {result}"
            except ValueError as e:
                return str(e)

        return (
            "Usage: /services [list] | /services start <name> "
            "| /services stop <name> | /services enable <name> "
            "| /services disable <name>"
        )

    def _format_services_list(self) -> str:
        """Format service status for display."""
        statuses = self.service_manager.status()
        if not statuses:
            return "No services registered."

        lines = ["Services:"]
        for svc in statuses:
            status = svc["status"]
            name = svc["name"]
            desc = svc.get("description", "")
            uptime = svc.get("uptime_seconds")
            error = svc.get("error")

            status_str = f"[{status}]"
            if uptime is not None:
                status_str += f" uptime {uptime}s"
            if error:
                status_str += f" error: {error}"

            lines.append(f"  {name}: {desc} {status_str}")

        return "\n".join(lines)
