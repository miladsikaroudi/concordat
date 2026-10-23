"""τ-bench's command line, with ``concordat_agent`` registered alongside its own agents.

uv run --project tau2-bench python run.py run --domain airline --agent concordat_agent ...
"""

from concordat_tau import create_concordat_agent
from tau2.cli import main
from tau2.registry import registry

registry.register_agent_factory(create_concordat_agent, "concordat_agent")

if __name__ == "__main__":
    main()
