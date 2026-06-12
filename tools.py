import subprocess
import sys
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class _CodeInput(BaseModel):
    code: str = Field(description="Python3 code to execute. ALWAYS use print() to show results.")


class LocalPythonTool(BaseTool):
    """Runs Python code in the local environment without Docker or extra parameters."""

    name: str = "Python Code Executor"
    description: str = (
        "Executes Python3 code locally and returns stdout+stderr. "
        "Use print() to display results. All installed packages are available."
    )
    args_schema: type[BaseModel] = _CodeInput

    def _run(self, code: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output.strip() or "(no output)"


code_tool = LocalPythonTool()
