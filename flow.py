import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

from crews.analyst_crew import analyst_crew
from crews.scientist_crew import scientist_crew

load_dotenv()

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CREW1_OUTPUTS = [
    "clean_data.csv",
    "eda_report.html",
    "insights.md",
    "dataset_contract.json",
]

CREW2_OUTPUTS = [
    "features.csv",
    "model.pkl",
    "evaluation_report.md",
    "model_card.md",
]

# ── flow state ──────────────────────────────────────────────────────────────
class WalmartState(BaseModel):
    analyst_status:   str   = "pending"   # pending | completed | failed
    scientist_status: str   = "pending"   # pending | completed | failed
    errors:           list  = []          # accumulated error messages
    start_time:       float = 0.0         # unix timestamp set at pipeline start


# ════════════════════════════════════════════════════════════════════════════
# FLOW
# ════════════════════════════════════════════════════════════════════════════
class WalmartFlow(Flow[WalmartState]):

    # ── step 1: run Crew 1 ──────────────────────────────────────────────────
    @start()
    def run_analyst_crew(self):
        self.state.start_time = time.time()
        print("\n" + "═" * 60)
        print(f"🚀 [FLOW] Starting Walmart AI Pipeline — {datetime.now().strftime('%H:%M:%S')}")
        print("═" * 60)
        print("\n📊 [CREW 1] Running Data Analyst Crew...")

        try:
            analyst_crew.kickoff()
            self.state.analyst_status = "completed"
            elapsed = round(time.time() - self.state.start_time, 1)
            print(f"✓  [CREW 1] Finished in {elapsed}s")
        except Exception as e:
            self.state.analyst_status = "failed"
            self.state.errors.append(f"Crew 1 error: {str(e)}")
            raise RuntimeError(
                f"\n❌ [CREW 1] Failed with error:\n   {e}\n"
                "   Fix the Analyst Crew before re-running."
            ) from e

    # ── step 2: validate Crew 1 outputs ─────────────────────────────────────
    @listen(run_analyst_crew)
    def validate_analyst_outputs(self):
        print("\n🔍 [VALIDATION] Checking Crew 1 outputs...")

        missing = []
        for filename in CREW1_OUTPUTS:
            path = ARTIFACTS_DIR / filename
            if path.exists():
                print(f"   ✓ {filename}")
            else:
                print(f"   ✗ {filename}  ← MISSING")
                missing.append(filename)

        if missing:
            self.state.analyst_status = "failed"
            self.state.errors.extend(missing)
            raise FileNotFoundError(
                f"\n❌ [VALIDATION] Crew 1 outputs incomplete.\n"
                f"   Missing files: {missing}\n"
                "   Fix the Analyst Crew before re-running."
            )

        print("✓  [VALIDATION] All Crew 1 files present — proceeding to Crew 2\n")

    # ── step 3: run Crew 2 ──────────────────────────────────────────────────
    @listen(validate_analyst_outputs)
    def run_scientist_crew(self):
        print("🔬 [CREW 2] Running Data Scientist Crew...")
        crew2_start = time.time()

        try:
            scientist_crew.kickoff()
            self.state.scientist_status = "completed"
            elapsed = round(time.time() - crew2_start, 1)
            print(f"✓  [CREW 2] Finished in {elapsed}s")
        except Exception as e:
            self.state.scientist_status = "failed"
            self.state.errors.append(f"Crew 2 error: {str(e)}")
            raise RuntimeError(
                f"\n❌ [CREW 2] Failed with error:\n   {e}\n"
                "   Fix the Scientist Crew before re-running."
            ) from e

    # ── step 4: validate Crew 2 outputs ─────────────────────────────────────
    @listen(run_scientist_crew)
    def validate_scientist_outputs(self):
        print("\n🔍 [VALIDATION] Checking Crew 2 outputs...")

        missing = []
        for filename in CREW2_OUTPUTS:
            path = ARTIFACTS_DIR / filename
            if path.exists():
                print(f"   ✓ {filename}")
            else:
                print(f"   ✗ {filename}  ← MISSING")
                missing.append(filename)

        if missing:
            self.state.scientist_status = "failed"
            self.state.errors.extend(missing)
            raise FileNotFoundError(
                f"\n❌ [VALIDATION] Crew 2 outputs incomplete.\n"
                f"   Missing files: {missing}\n"
                "   Fix the Scientist Crew before re-running."
            )

        print("✓  [VALIDATION] All Crew 2 files present\n")

    # ── step 5: final summary ───────────────────────────────────────────────
    @listen(validate_scientist_outputs)
    def pipeline_done(self):
        total = round(time.time() - self.state.start_time, 1)

        print("═" * 60)
        print("🎉 [DONE] Pipeline completed successfully!")
        print(f"   Total time  : {total}s")
        print(f"   Artifacts   : {ARTIFACTS_DIR}")
        print("\n   Crew 1 outputs:")
        for f in CREW1_OUTPUTS:
            print(f"     • {f}")
        print("\n   Crew 2 outputs:")
        for f in CREW2_OUTPUTS:
            print(f"     • {f}")
        print("═" * 60 + "\n")


# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flow = WalmartFlow()
    flow.kickoff()
