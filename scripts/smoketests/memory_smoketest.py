#!/usr/bin/env python3
"""
Memory smoketest — semantic retrieval across multiple memory files.

Writes 4 memory files on distinct topics directly into the workspace after
seeding, then runs memory_search and memory_get with a question whose answer
appears in exactly one paragraph of one file. Prints the question and
retrieved answer for both tools and asserts the right content was found.

Requires OPENROUTER_API_KEY to be set.

Usage:
    python3 smoketests/memory_smoketest.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent.parent))

sys.path.insert(0, str(_REPO_ROOT))

from runtime.config import Config
from sandbox.workspace import SandboxGuard, init_workspace
from analysis_tools.self_improvement.tracker import EvolutionTracker
from agent.tools.core.created_tools import load_created_tools
from agent.tools.core.filesystem import make_filesystem_tools
from agent.tools.core.memory import make_memory_tools
from agent.tools.core.self_improve import make_self_improve_tools
from agent.tools.registry import ToolRegistry
from agent.session import Session

_CONTROLLER_SURFACE = _REPO_ROOT / "agent" / "seeded_files"

config = Config()
artifacts_root = Path(config.artifacts_dir)
env_assets_dir = Path(config.env_assets_dir)
timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
run_dir = artifacts_root / f"memory_smoketest_{timestamp}"
workspace_dir = Path(config.workspace_dir) if config.workspace_dir else run_dir / "workspace"
log_dir = run_dir / "logs" / "event_logs"
checkpoints_dir = run_dir / "checkpoints"
analysis_dir = run_dir / "analysis"
for path in (workspace_dir, log_dir, checkpoints_dir, analysis_dir):
    path.mkdir(parents=True, exist_ok=True)

# ── Require live API key ──────────────────────────────────────────────────────
api_key = os.environ.get(config.api_key_env, "").strip()
if not api_key:
    print(f"SKIP: {config.api_key_env} not set — this smoketest requires a live API call")
    sys.exit(0)

# ── Workspace init ────────────────────────────────────────────────────────────
print("\n=== 1. Workspace initialisation ===")
init_workspace(env_assets_dir, workspace_dir, controller_surface_dir=_CONTROLLER_SURFACE)
print(f"  ✓  workspace ready at {workspace_dir}")

# ── Wire up components ────────────────────────────────────────────────────────
print("\n=== 2. Session setup ===")
guard = SandboxGuard(workspace_dir)
tracker = EvolutionTracker(log_dir / "memory_smoketest_events.jsonl")
registry = ToolRegistry()

for name, handler, desc, params in make_filesystem_tools(guard):
    registry.register(name, handler, desc, params)
for name, handler, desc, params in make_memory_tools(workspace_dir):
    registry.register(name, handler, desc, params)
for name, handler, desc, params in load_created_tools(workspace_dir):
    registry.register(name, handler, desc, params)
for name, handler, desc, params in make_self_improve_tools(workspace_dir, tracker, registry):
    registry.register(name, handler, desc, params)
registry.register_checkpoint()

session = Session(config, registry, tracker, workspace_dir)
session.init()
print("  ✓  session initialised")

# ── Write 4 memory files on distinct topics ───────────────────────────────────
print("\n=== 3. Writing memory files ===")

today = datetime.date.today().isoformat()

MEMORY_FILES = {
    f"memory/{today}-fitness.md": """\
# Fitness Log — Q2 2026

## Weekly Training Summary
April saw a strong return to structured training after a rest week in late March.
Weekly mileage averaged 28 km across four runs, with a long run on Sundays capped
at 12 km. Heart rate data from the wearable shows aerobic efficiency improving,
with zone-2 sessions consistently staying below 145 bpm. The April 6th run was
particularly strong — 10 km at a 5:12 pace, which is a new personal best for that
distance on a flat course. Cadence averaged 174 spm, up from 168 spm in March,
suggesting improved running economy from the drills added in week two.

## Strength Training
Two gym sessions per week focused on posterior chain work: Romanian deadlifts,
hip thrusts, and single-leg exercises. Deadlift working weight progressed from
80 kg to 87.5 kg over the month without any form breakdown noted. Hip thrust volume
was kept at 4×10 at 100 kg. The single-leg Romanian deadlift remains a weak point —
balance on the left side noticeably worse than the right; added ankle mobility work
to the warm-up routine starting April 20th.

## Recovery and Sleep
Sleep quality averaged 7.2 hours with a sleep score of 81, slightly above the 90-day
baseline of 78. The two poorest nights (scores of 61 and 67) both followed evenings
with screen time past 23:00. Resting heart rate trended down from 52 bpm at the start
of April to 49 bpm by the 28th, consistent with improving cardiovascular fitness.
One rest day was taken mid-week each week; no unplanned rest days due to fatigue or
soreness, which suggests the training load is well-calibrated.

## Nutrition Notes
Protein intake was tracked at roughly 145 g/day, in line with the 1.8 g/kg target.
Hydration improved after adding a mid-morning reminder — average daily water intake
rose from 1.9 L to 2.4 L. Caloric surplus of ~200 kcal on training days was
maintained to support the strength phase. Pre-workout nutrition settled on a banana
and 20 g whey 45 minutes before sessions; post-workout: Greek yoghurt with berries.
No GI issues reported during long runs after removing the pre-run coffee experiment.

## Goals for May
Increase long run to 14 km by end of May. Add a tempo session (threshold pace,
20 min) once per week. Retest deadlift max in the final week of May. Continue
sleep hygiene improvements — hard stop on screens at 22:30.
""",

    f"memory/{today}-finance.md": """\
# Financial Review — April 2026

## Income
Direct deposit of $5,200 received on April 1st, consistent with monthly take-home
after pre-tax 401(k) contribution of $800 and health insurance deduction of $210.
No bonus, freelance income, or other variable income this month. Year-to-date
gross income stands at $20,800 against a plan of $21,000 — slightly behind due to
a payroll timing issue in January that has since been corrected.

## Fixed Expenses
Rent of $1,850 debited on April 3rd. Utilities came to $94 (electric $61, water $33),
which is lower than the $115 March figure thanks to milder temperatures. Internet and
phone combined at $118. Streaming and software subscriptions totalled $47 — cancelled
one unused service (Headspace, $13/month) effective May 1st. Renter's insurance
auto-paid at $22. Total fixed expenses: $2,131.

## Discretionary Spending
Dining out accounted for $340 in April, up from $210 in March — mostly due to a team
dinner on April 14th ($95) and two weekend brunches. Grocery spending came in at $384,
slightly under the $400 monthly budget target. Entertainment was $62 (two cinema
tickets and one concert). Clothing: $0. Personal care: $38. Coffee shops: $55,
which is higher than intended — will try batch-brewing at home for weekday mornings.

## Savings and Investments
Transferred $800 to the high-yield savings account on April 15th, bringing the
emergency fund to $14,200, which covers 2.7 months of expenses at the current burn
rate. The brokerage auto-investment of $300 into a total-market index fund executed
on April 7th. Current brokerage balance: $23,450, up from $21,900 at end of March,
boosted by a 6.2% market return on existing holdings. 401(k) balance not checked
this month — will review at end of Q2.

## Debt
No credit card balance carried — statement balance of $892 paid in full on April 18th.
No student loans or car payments. Remaining balance on the 0% APR laptop financing
plan: $240, final payment due June 1st.

## Outstanding Items and May Outlook
The gym membership annual renewal of $540 is due May 1st — will pay in full rather
than month-to-month to save $108 annually. Insurance premium review scheduled for
mid-May; current monthly cost is $187. Expecting a $200 utility refund from the
building in May for overpayment in Q1. May travel spending will be elevated due to
the Asheville trip ($490 estimated for hotel, fuel, and dining).
""",

    f"memory/{today}-travel.md": """\
# Travel Planning — Summer 2026

## Asheville Trip (May 23–26)
A long weekend in Asheville, NC is confirmed. Hotel booked at the Inn on Biltmore
Estate at $245/night for three nights ($735 total before taxes). No flights needed —
driving approximately 4.5 hours from home. Car was serviced in March so no concerns
there. Restaurant reservation made at Cúrate for Saturday dinner (7:30 PM, party of
two, confirmation #CR-48821). Sunday brunch target is Tupelo Honey — no reservation
required. Plan to hike the Black Balsam Knob trail on Sunday morning (moderate, 4.2
miles round trip, 1,100 ft elevation gain). Will pack layers — Asheville in late May
can be 15°C in the mornings even at lower elevations.

## Japan Trip Research (September 2026)
Researching a two-week trip to Japan targeting the third week of September, which
avoids the peak summer heat and precedes autumn foliage crowds. Flights from Dulles
(IAD) to Tokyo Haneda (HND) are currently $1,080–$1,150 round trip on ANA non-stop.
Setting a price alert at $950 — will book if it drops. A JR Pass for 14 days costs
approximately $500 and covers all shinkansen travel between cities. Itinerary sketch:
Tokyo (4 nights), Hakone day trip, Kyoto (4 nights), Osaka (3 nights), Hiroshima/
Miyajima day trip, final night in Tokyo. Accommodation research ongoing — targeting
a mix of business hotels ($80–$110/night) and one traditional ryokan in Kyoto (~$180).

## Japan Budget Estimate
Flights: $1,100. JR Pass: $500. Accommodation (14 nights avg $100): $1,400.
Food ($60/day): $840. Activities and entry fees: $300. Local transport (IC card): $120.
Souvenirs and miscellaneous: $250. Total estimate: $4,510. Will start a dedicated
Japan savings bucket ($300/month from May through August = $1,200) to supplement
existing travel fund balance of $3,100.

## Packing Strategy
Carry-on only confirmed for Asheville — REI Flash 22 pack is sufficient for three
nights. Japan will require a checked bag (Osprey Farpoint 40) given 14 days and the
need for layers across varying climates. Travel insurance quotes pending — comparing
World Nomads (adventure tier, $180 for two weeks) vs. Allianz (AllTrips Premier,
$210). Global Entry renewal due in August; TSA Pre-check remains active. Will
schedule Global Entry appointment in June to avoid any gap.

## Other Trips on the Radar
A potential long weekend in Montreal in late October — would require flights (~$280
round trip from nearby airport). No bookings made yet; gauging interest from friends
first. A family visit over Thanksgiving is likely — will book flights by end of June
to get ahead of price increases.
""",

    f"memory/{today}-work.md": """\
# Work Notes — April 2026

## Q2 Roadmap and Priorities
The Q2 roadmap was finalised in the April 2nd all-hands. Three major initiatives
were confirmed: platform reliability (P0), the new onboarding redesign (P1), and
internal tooling improvements (P2). The reliability initiative is owned by the
infrastructure team with a concrete target of reducing p99 API latency below 200 ms
by end of May. Current p99 sits at 340 ms, primarily driven by a slow database query
path in the recommendations service that was identified in the March post-mortem.
The onboarding redesign is a cross-functional effort with design and growth leading;
engineering scope is estimated at six weeks of backend work and four weeks of frontend.

## Team Updates
Two new engineers — Priya Nair (backend, distributed systems background) and
Marcus Webb (frontend, React/TypeScript focus) — joined on April 7th. Onboarding
is being handled through the buddy system; Priya is paired with David Park and
Marcus with Emma Wilson. First independent task assignments expected by April 21st.
The team is now at 11 engineers, which is at the upper end of the planned headcount
for H1. Performance review cycle opens May 1st with self-assessments due May 15th;
manager reviews due May 29th; calibration sessions scheduled for June 3rd–5th.

## Key Decision: Authentication Migration
After a three-week evaluation comparing Auth0, Okta, and an in-house OAuth 2.0
implementation, the team decided on April 15th to build the in-house solution using
OAuth 2.0 with PKCE for the SPA client flow and client credentials for service-to-
service calls. The decision memo cited cost (Auth0 at current scale would run ~$2,400/
month), control over token lifecycle, and the team's existing familiarity with the
JWT library already in use. The migration is scoped for Q2, with a feature flag
rollout starting June 1st targeting 5% of traffic, scaling to 50% by June 15th, and
full cutover by June 30th. The legacy session-cookie system will be deprecated but
not removed until Q3 to allow rollback if edge cases emerge in production.

## Engineering Process Changes
Starting April 14th, all PRs require two approvals instead of one for changes
touching the payments or authentication paths. This was a recommendation from the
security review completed in late March. Additionally, load testing is now a
required step before any service that handles >1,000 RPM is deployed to production.
The CI pipeline was updated to include a Lighthouse score check for all frontend
changes — any PR dropping the performance score below 85 is blocked from merging.

## Action Items and Upcoming Deadlines
- Finalise API contract for auth migration: April 22nd (owner: David Park)
- Schedule load testing for reliability initiative: week of April 28th
- Send performance review kickoff email to team: April 30th (owner: me)
- First auth migration PR open for review: May 5th
- Onboarding redesign wireframes sign-off: May 9th
- Mid-Q2 reliability check-in with CTO: May 14th
""",
}

# The needle: only in the work file, only in the Key Decision paragraph
SEARCH_QUESTION = "What authentication protocol did the team decide to use for the SPA client flow?"
NEEDLE_PHRASE = "oauth 2.0 with pkce"

for rel_path, content in MEMORY_FILES.items():
    path = workspace_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  ✓  wrote {rel_path} ({len(content)} chars)")

# ── memory_search ─────────────────────────────────────────────────────────────
print("\n=== 4. memory_search ===")
print(f"\n  Question: {SEARCH_QUESTION}\n")

result = registry.dispatch("memory_search", {"query": SEARCH_QUESTION, "limit": 1})
assert result.success, f"FAIL: memory_search returned an error — {result.output}"

data = json.loads(result.output)
assert data["count"] > 0, (
    f"FAIL: memory_search returned no results for query:\n  '{SEARCH_QUESTION}'"
)

answer_snippet = data["results"][0]
print("  Retrieved text:")
print()
for line in answer_snippet.splitlines():
    print(f"    {line}")
print()

assert NEEDLE_PHRASE in answer_snippet.lower(), (
    f"FAIL: expected needle '{NEEDLE_PHRASE}' not found in memory_search result.\n"
    f"Got:\n{answer_snippet}"
)
print(f"  ✓  needle phrase found: '{NEEDLE_PHRASE}'")

# ── memory_get ────────────────────────────────────────────────────────────────
print("\n=== 5. memory_get ===")
print(f"\n  Question: {SEARCH_QUESTION}\n")

result2 = registry.dispatch("memory_get", {"query": SEARCH_QUESTION, "limit": 1})
assert result2.success, f"FAIL: memory_get returned an error — {result2.output}"

data2 = json.loads(result2.output)
assert data2["count"] > 0, "FAIL: memory_get returned no results"

answer_snippet2 = data2["results"][0]
print("  Retrieved text:")
print()
for line in answer_snippet2.splitlines():
    print(f"    {line}")
print()

assert NEEDLE_PHRASE in answer_snippet2.lower(), (
    f"FAIL: needle not found in memory_get result.\nGot:\n{answer_snippet2}"
)
print(f"  ✓  needle phrase found: '{NEEDLE_PHRASE}'")

print(f"\n=== Memory smoketest passed ===")
print(f"    Run dir: {run_dir}")
