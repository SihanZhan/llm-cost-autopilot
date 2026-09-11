"""Phase 2: build the hand-labeled training set for the complexity classifier.

Each prompt below was written and tier-assigned by hand against the
definitions in ``classifier.tiers`` — deliberately varied across domains and
phrasing so the classifier has to learn real signal (verb choice, constraint
count, structure requests, reasoning demands) rather than memorize a
template. Running this module writes ``labeled_prompts.jsonl`` next to it.

Usage:
    python -m classifier.data.build_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

from classifier.tiers import TIER_COMPLEX, TIER_MODERATE, TIER_SIMPLE

OUT_FILE = Path(__file__).parent / "labeled_prompts.jsonl"

# Tier 1 — simple: reformatting, extraction, basic Q&A. Short, deterministic answers.
TIER1_PROMPTS = [
    "Extract the phone number from this text and return only the number: \"Call the front desk anytime at 555-201-4488 for parking passes.\"",
    "Convert this list into a JSON array of strings, nothing else: cereal, milk, bananas, honey",
    "What is the capital of France? Answer with just the city name.",
    "Reformat this date as YYYY-MM-DD and output only the result: July 4, 2025",
    "Convert 5 miles to kilometers. Output only the number, rounded to one decimal.",
    "Translate the word \"hello\" into Spanish. Output only the translated word.",
    "Capitalize the first letter of each word in this title and return only the result: \"the great gatsby\"",
    "Sort these numbers in ascending order and return them as a comma-separated list: 42, 7, 19, 3, 88",
    "Rename the variable \"usr\" to \"user\" in this line and return only the new line: `usr = get_profile(id)`",
    "How many items are in this list? Answer with just the number: pen, notebook, eraser, ruler, stapler",
    "Extract the order number from this text and return only the number: \"Your package (Order #A29184) has shipped.\"",
    "Lowercase this string and return only the result: \"PLEASE RESPOND ASAP\"",
    "Uppercase this string and return only the result: \"quarterly report draft\"",
    "Replace every instance of \"cat\" with \"dog\" in this sentence and return only the result: \"The cat sat by the cat door.\"",
    "Count the number of vowels in this word and return only the number: \"encyclopedia\"",
    "Copy this sentence exactly as written, with no changes: \"Meet me at the corner at noon.\"",
    "Repeat the word \"echo\" three times, separated by spaces, and return only that.",
    "Parse the username from this email address and return only the username: \"j.rivera88@mailbox.example\"",
    "Strip the leading and trailing whitespace from this string and return only the result: \"   hello world   \"",
    "Join these words into a single sentence separated by spaces, nothing else: \"The\", \"sun\", \"set\", \"early\"",
    "Split this sentence into individual words as a JSON array: \"Turn left at the second light\"",
    "Round 17.386 to two decimal places and output only the number.",
    "What is the boiling point of water in Celsius at sea level? Answer with just the number.",
    "Extract the zip code from this address and return only the zip code: \"412 Birch Ave, Denver, CO 80203\"",
    "Reformat this phone number as (XXX) XXX-XXXX and output only the result: 3035557821",
    "Convert 2 cups to tablespoons. Output only the number.",
    "What's the chemical symbol for gold? Answer with just the symbol.",
    "Extract the hashtags from this caption and return them as a JSON array: \"Beach day! #sunset #ocean #vacation\"",
    "Reformat this name as \"Last, First\" and output only the result: \"Priya Natarajan\"",
    "Convert this temperature from Fahrenheit to Celsius: 98.6. Output only the number, rounded to one decimal.",
    "What year did the first iPhone release? Answer with just the year.",
    "Extract the URL from this text and return only the URL: \"You can register at https://example.com/signup before Friday.\"",
    "Reformat this list of names into \"Last, First\" format, one per line, nothing else: John Smith, Maria Garcia, Tom Lee",
    "Convert this binary number to decimal and output only the result: 101101",
    "What is the plural of \"cactus\"? Answer with just the word.",
    "Extract the invoice total from this text and return only the amount: \"Total due: $482.19, payable within 30 days.\"",
    "Reformat this list as a numbered list, nothing else: milk, eggs, bread",
    "Convert 3 kilograms to pounds. Output only the number, rounded to one decimal.",
    "What is the freezing point of water in Fahrenheit? Answer with just the number.",
    "Extract the flight number from this text and return only the flight number: \"Your flight, DL 2049, departs at 6:15 AM from gate B12.\"",
    "Translate \"good morning\" into French. Output only the translated phrase.",
    "Reformat this time as 24-hour format and output only the result: 9:45 PM",
    "Extract the discount code from this email and return only the code: \"Use code SAVE20 at checkout for 20% off your order.\"",
    "Convert this hex color code to its RGB values, output only the three numbers separated by commas: #FF5733",
    "What is the square root of 144? Answer with just the number.",
    "Reformat this address into a single line and return only the result: \"123 Maple St, Apt 4B, Boston, MA 02118\"",
    "Extract the tracking number from this text and return only the number: \"Tracking: 1Z999AA10123456784, expected Thursday.\"",
    "Convert 12 tablespoons to cups. Output only the number.",
    "What is the abbreviation for \"Doctor\"? Answer with just the abbreviation.",
    "Reformat these initials with periods and return only the result: \"jrr\"",
    "Extract the meeting time from this message and return only the time: \"Quick sync tomorrow at 2:30pm, keep it short.\"",
    "Convert 100 USD to a rough estimate in EUR at a 0.92 rate. Output only the number.",
    "What is 15% of 200? Answer with just the number.",
    "Reformat this username by removing all numbers and return only the result: \"skywalker42_luke07\"",
    "Extract the room number from this text and return only the number: \"Conference is in Room 314B this year.\"",
    "Convert this Roman numeral to a regular number and output only the result: XLII",
    "What is the past tense of \"run\"? Answer with just the word.",
    "Reformat this list alphabetically and return it as a comma-separated list: banana, apple, cherry, date",
    "Extract the coupon expiration date from this text and return only the date: \"Offer valid through 12/31/2026, no exceptions.\"",
    "Convert 250 grams to ounces. Output only the number, rounded to one decimal.",
    "What is the currency used in Japan? Answer with just the name.",
    "Reformat this social handle to remove the \"@\" symbol and return only the result: \"@traveldiaries\"",
    "Extract the confirmation number from this booking text and return only the number: \"Your reservation is confirmed under number RES-88213.\"",
    "Convert 7 feet to meters. Output only the number, rounded to two decimals.",
    "What is the opposite of \"ascend\"? Answer with just the word.",
    "Reformat this list into title case and return only the result, one per line: \"the hobbit\", \"1984\", \"dune\"",
    "Extract the promo percentage from this text and return only the number: \"Members get an extra 15% off this weekend only.\"",
    "Convert 30 minutes to seconds. Output only the number.",
    "What is the SI unit for force? Answer with just the unit name.",
    "Reformat this string by removing all punctuation and return only the result: \"Wait... what?! Really?\"",
    "Extract the appointment date from this text and return only the date: \"See you at the dentist on September 22nd at 10am.\"",
    "Convert this fraction to a decimal and output only the number: 3/8",
    "What is the plural of \"child\"? Answer with just the word.",
    "Reformat this list of tags by removing duplicates and return as a comma-separated list: red, blue, red, green, blue",
]

# Tier 2 — moderate: summarize, classify, compare, describe, explain, outline.
TIER2_PROMPTS = [
    "Summarize this paragraph in one sentence: \"The city council approved a new bike lane network last week. Supporters say it will cut commute times and improve safety, while some business owners worry about lost parking. Construction is expected to begin in the spring.\"",
    "Classify the sentiment of this review as positive, negative, or neutral, and reply with only that word: \"The soup was lukewarm but the service made up for it.\"",
    "List three advantages and three disadvantages of switching my blog from WordPress to a static site generator. Use two short bulleted lists.",
    "Summarize this email thread in two sentences: \"Subject: Budget review. Hi team, quick note that Q3 numbers came in under target, mostly due to the delayed product launch. Let's regroup Monday to discuss reallocation. — Dana\"",
    "Classify this support ticket as \"billing\", \"technical\", or \"account access\", and reply with only that label: \"I can't log in even after resetting my password twice.\"",
    "Compare renting versus buying a car for someone who drives about 8,000 miles a year. Give two points for each side.",
    "Summarize the plot of this short synopsis in one sentence: \"A retired detective is pulled back into one last case when a cold file resurfaces, forcing her to confront the partner she blamed for it going unsolved.\"",
    "Classify this news headline by topic — \"sports\", \"politics\", \"technology\", or \"health\" — and reply with only that word: \"City Hospital Opens New Pediatric Wing\"",
    "Describe the main differences between a checking account and a savings account in three bullet points.",
    "Summarize this meeting transcript excerpt in one sentence: \"So the main blocker is still the vendor contract — legal flagged two clauses on data retention. Marketing's ready to launch as soon as that's resolved.\"",
    "Classify this product review as positive, negative, or neutral, and reply with only that word: \"It works fine, does what it says, nothing special.\"",
    "Outline the steps to set up a home Wi-Fi network for a first-time user, in four bullet points.",
    "Summarize this recipe's key steps in two sentences: \"Preheat the oven to 375°F. Season the chicken thighs and sear them skin-side down for 5 minutes. Transfer to the oven and roast for 25 minutes until the skin is crisp and the juices run clear.\"",
    "Classify this email as \"urgent\" or \"not urgent\", and reply with only that word: \"No rush at all, just checking in whenever you get a chance.\"",
    "Compare a 15-year mortgage and a 30-year mortgage for someone prioritizing lower monthly payments. Two points each.",
    "Summarize this customer complaint in one sentence: \"I ordered the blue one but received red, and when I called support they said I'd have to pay for return shipping myself, which doesn't seem right given it was their mistake.\"",
    "Group these words into two categories, \"fruit\" and \"vegetable\", and list each group: carrot, apple, spinach, banana, broccoli, mango",
    "Explain the difference between a gross salary and a net salary in two sentences.",
    "Rank these three laptops by battery life based on this description, best to worst, and just list the order: \"Laptop A: 10 hours. Laptop B: 14 hours. Laptop C: 8 hours.\"",
    "Summarize this project status update in one sentence: \"Backend is done and tested, frontend is about 70% complete, and we're still waiting on the design team for the final icon set before we can ship.\"",
    "Classify this workplace message as \"positive feedback\", \"negative feedback\", or \"neutral\", and reply with only that phrase: \"Good catch on the bug, that would've been a mess in production.\"",
    "Identify the main cause described in this passage in one sentence: \"Traffic on Route 9 has worsened significantly since the new outlet mall opened, adding an estimated 15 minutes to the average commute.\"",
    "Describe three pros and three cons of adopting a puppy versus an adult dog.",
    "Summarize this doctor's note in one sentence: \"Patient presents with mild seasonal allergy symptoms, advised to continue over-the-counter antihistamines and follow up if symptoms worsen or persist beyond two weeks.\"",
    "Classify this tweet's tone as \"sarcastic\", \"sincere\", or \"neutral\", and reply with only that word: \"Oh great, another Monday meeting that could've been an email.\"",
    "Compare working from home versus working from an office for someone with young kids. Give two points for each.",
    "Summarize the key takeaway from this study abstract in one sentence: \"Participants who took short walking breaks every hour reported significantly less afternoon fatigue than those who remained seated, though the effect on overall productivity was inconclusive.\"",
    "Explain why a credit score might drop after paying off a loan, in two sentences.",
    "Outline three steps for onboarding a new hire on their first day.",
    "Classify this restaurant review as positive, negative, or neutral, and reply with only that word: \"Great food, but we waited 45 minutes for a table despite having a reservation.\"",
    "Summarize this incident report in one sentence: \"At approximately 2:15pm, a minor water leak was reported near the second-floor break room; maintenance was dispatched and resolved the issue within the hour with no equipment damage.\"",
    "Describe the difference between a personal loan and a home equity line of credit in two to three sentences.",
    "Group these expenses into \"essential\" and \"discretionary\": rent, streaming subscriptions, groceries, dining out, electricity, concert tickets",
    "Summarize this teacher's feedback in one sentence: \"Your essay has a strong thesis, but the second paragraph strays from the main argument — tightening that section would make the whole piece more persuasive.\"",
    "Classify this job listing's seniority level as \"entry\", \"mid\", or \"senior\", and reply with only that word: \"3+ years of experience required, some mentorship of junior team members expected.\"",
    "Compare paper books and e-readers for someone who travels frequently. Two points each.",
    "Explain in two sentences why a website might load slowly on mobile but fine on desktop.",
    "Summarize this travel itinerary note in one sentence: \"Day one is arrival and rest, day two is the museum district and a walking food tour, and day three is a half-day trip to the coast before the evening flight home.\"",
    "Identify the tone of this cover letter opening as \"confident\", \"hesitant\", or \"neutral\", and reply with only that word: \"I believe my five years leading cross-functional teams make me well-suited for this role.\"",
    "Describe two advantages and two disadvantages of a four-day work week.",
    "Summarize this warranty clause in one sentence: \"Coverage applies only to manufacturing defects reported within 12 months of purchase and does not extend to accidental damage or normal wear and tear.\"",
    "Classify this social media comment as \"spam\", \"genuine feedback\", or \"complaint\", and reply with only that phrase: \"This is exactly the update I've been waiting for, thank you!\"",
    "Compare index funds and individual stock picking for a beginner investor. Two points each.",
    "Summarize this recipe review in one sentence: \"Followed it exactly and the cake came out a little dry, though the frosting recipe was excellent and I'll definitely reuse that part.\"",
    "Outline three considerations when choosing a college major.",
    "Explain the difference between \"affect\" and \"effect\" in one or two sentences with an example of each.",
    "Classify this weather alert's severity as \"low\", \"moderate\", or \"severe\", and reply with only that word: \"Expect scattered light showers through the afternoon with no significant wind.\"",
    "Summarize this landlord notice in one sentence: \"Water will be shut off building-wide from 9am to 1pm on Thursday for scheduled pipe maintenance; no action needed from tenants.\"",
    "Compare a standing desk and a traditional desk for someone with lower back pain. Two points each.",
    "Describe two reasons a startup might choose to bootstrap instead of raising venture capital.",
    "Summarize this app update changelog in one sentence: \"This release fixes the crash on login, improves photo upload speed, and adds dark mode support for all screens.\"",
    "Classify this customer email's urgency as \"high\", \"medium\", or \"low\", and reply with only that word: \"My card was charged twice for the same order and I need this resolved before it affects my account balance.\"",
    "Explain in two sentences the difference between a resume and a CV.",
    "Outline the main steps to file a warranty claim based on this policy summary: \"Claims must include proof of purchase and photos of the defect, submitted through the online portal within 30 days.\"",
    "Compare a gas stove and an induction stove for someone who cooks daily. Two points each.",
    "Summarize this HR policy update in one sentence: \"Starting next quarter, employees may carry over up to five unused vacation days into the following year, up from zero previously.\"",
    "Classify this online comment as \"constructive criticism\" or \"just complaining\", and reply with only that phrase: \"The checkout flow is confusing — maybe combine steps 2 and 3?\"",
    "Describe two pros and two cons of meal prepping for the week versus cooking daily.",
    "Summarize this insurance claim update in one sentence: \"Your claim has been approved for partial coverage; the adjuster's report cited pre-existing wear on the affected part, which reduced the payout by 20%.\"",
    "Explain why two students might get different grades on similar essays, in two to three sentences.",
    "Compare a fixed-rate and an adjustable-rate mortgage for someone planning to move in three years. Two points each.",
    "Outline three factors to weigh when picking a moving company.",
    "Summarize this software release note in one sentence: \"Version 4.2 addresses the memory leak reported by several users on the forums and adds an optional auto-save feature, though initial testing shows a small increase in startup time.\"",
    "Classify this voicemail transcript's purpose as \"sales\", \"support\", or \"personal\", and reply with only that word: \"Hey, it's your cousin, just calling to see if you're free for dinner Sunday.\"",
    "Describe two differences between a manager and a mentor.",
    "Summarize this product recall notice in one sentence: \"The manufacturer has issued a voluntary recall on units produced between March and June due to a battery overheating risk, and is offering free replacements.\"",
    "Compare studying with flashcards versus studying by writing summaries for retaining new vocabulary. Two points each.",
    "Explain in two sentences why a router might need to be restarted periodically.",
    "Outline three things to check before signing an apartment lease.",
    "Summarize this performance review excerpt in one sentence: \"Strong technical output this quarter, but communication with stakeholders during the outage could have been faster and more proactive.\"",
    "Classify this text message as \"needs a reply\" or \"no reply needed\", and reply with only that phrase: \"Running 10 min late, see you soon!\"",
    "Compare a hybrid car and a fully electric car for someone with a long daily commute and no home charger. Two points each.",
    "Describe two reasons remote teams might rely more heavily on written documentation than in-office teams.",
    "Summarize this contractor's estimate note in one sentence: \"The quote covers materials and labor for the bathroom remodel but excludes any plumbing rerouting, which would be billed separately if needed.\"",
    "Explain the difference between a warranty and an insurance policy in two sentences.",
]

# Tier 3 — complex: multi-step reasoning, creative generation, nuanced judgment.
TIER3_PROMPTS = [
    "Two trains start 180 miles apart heading toward each other, one at 60 mph and the other at 40 mph. How long until they meet? Show your reasoning step by step.",
    "Write a four-line poem about a lighthouse. The poem must never mention the sea, water, waves, or the ocean.",
    "I have a $10,000 budget to reduce my monthly cloud hosting bill. Propose a prioritized three-step plan and justify why each step is ordered where it is.",
    "A tank is filled by pipe A in 6 hours and by pipe B in 4 hours. If both pipes are open together, how long will it take to fill the tank? Walk through your reasoning.",
    "Write a six-line poem about starting a new job, without using the words \"nervous\", \"excited\", or \"first\".",
    "My team of five has three months to migrate a legacy database with zero downtime allowed. Propose a step-by-step plan and explain the risks at each stage.",
    "If a recipe serves 4 and calls for 2.5 cups of flour, how much flour is needed for 10 people, and why might you round the final amount up rather than down? Explain your reasoning.",
    "Write a short story opening, no more than five sentences, about a locksmith who's afraid of locked doors, without directly stating the irony.",
    "Evaluate whether a small bakery should open a second location this year given rising rent costs but strong recent sales growth, and justify your recommendation.",
    "A car travels the first half of a trip at 50 mph and the second half at 70 mph. What is its average speed for the entire trip? Show each step.",
    "Write a four-line poem about autumn without using the words \"leaves\", \"orange\", \"fall\", or \"cold\".",
    "Design a three-phase rollout plan for a company switching all employees from a 5-day to a 4-day work week, and justify the order of the phases.",
    "I'm choosing between two job offers: one with higher pay but longer hours, and one with lower pay but full remote flexibility. Walk through the tradeoffs and recommend one, explaining your reasoning.",
    "A ladder 10 feet long leans against a wall with its base 6 feet from the wall. How high up the wall does the ladder reach? Show your work.",
    "Write a limerick about a chef who burns everything, without using the word \"burn\" or any of its forms.",
    "Propose a step-by-step plan for a first-time marathon runner training over 16 weeks, and explain why the phases are ordered the way they are.",
    "If I invest $5,000 at 6% annual interest compounded yearly, how much will I have after 8 years, and how does that compare to simple interest over the same period? Show your reasoning.",
    "Critique this argument and identify its main weakness: \"Since every successful startup founder I've read about dropped out of college, dropping out must improve your odds of success.\"",
    "Write a three-stanza poem about learning to ride a bike, without ever using the words \"bike\", \"wheel\", or \"pedal\".",
    "A farmer has 60 meters of fencing and wants to enclose the largest possible rectangular area against an existing wall, so only three sides need fencing. What dimensions maximize the area? Explain your reasoning.",
    "Recommend whether a small nonprofit should switch from paper mailers to an all-digital fundraising campaign, weighing donor demographics and cost, and justify your answer.",
    "Two runners start at the same point on a circular track 400 meters long, running in opposite directions at 5 m/s and 7 m/s. How long until they meet again? Show the steps.",
    "Write a haiku sequence, three haiku, about grief, without using the words \"loss\", \"gone\", \"death\", or \"sad\".",
    "Design a prioritized plan for a solo developer to add automated testing to a codebase that currently has none, and justify the order of steps given limited time.",
    "Forecast how a local coffee shop's revenue might change if a large chain opens two blocks away, and explain the factors driving your estimate.",
    "A pool can be filled by one hose in 5 hours and drained by a leak in 8 hours. If both are open at once, how long to fill the pool? Show your reasoning.",
    "Write a six-line poem about a chess match, without mentioning any chess piece by name.",
    "Propose a three-step plan to reduce food waste in a household of four, ordered from easiest to hardest to sustain, and justify the ordering.",
    "Brainstorm three original names for a mobile app that helps roommates split bills, and explain the reasoning behind each suggestion.",
    "If a car depreciates 15% in its first year and 10% of its remaining value each year after, what is it worth after 3 years starting from $30,000? Show each step.",
    "Write a short monologue, four to six sentences, from the perspective of an old bridge watching a city change, without using the word \"bridge\".",
    "Evaluate the tradeoffs between hiring a full-time employee versus a freelancer for a six-month project, and recommend one option with justification.",
    "A store marks up an item by 40% then offers a 20% discount on the marked-up price. Is the final price higher or lower than the original, and by how much? Show your reasoning.",
    "Design a plan for a first-time homeowner to prioritize repairs on a fixer-upper with a $15,000 budget, and justify what comes first.",
    "Write a four-line poem about insomnia without using the words \"sleep\", \"night\", \"awake\", or \"tired\".",
    "Hypothesize why a popular restaurant's online reviews might have dropped sharply after a change in ownership, and suggest what evidence would confirm or rule out each hypothesis.",
    "Two friends split a $240 restaurant bill unevenly because one ordered a $60 bottle of wine alone. Propose a fair way to split it and justify your reasoning.",
    "Write a short fable, five to seven sentences, in which a tortoise and a hare learn a lesson that is not \"slow and steady wins the race\".",
    "Recommend a prioritized order for a freelancer to pay down $8,000 across three debts with different interest rates, and justify the ordering.",
    "A photographer charges $200 for the first hour and $75 for each additional hour. If a client has a $500 budget, what's the maximum number of hours they can book, and what's left over? Show your work.",
    "Design a three-step plan to improve a small ecommerce store's checkout abandonment rate, and justify why each step comes before the next.",
    "Write a two-stanza poem about a city at 3am, without using the words \"quiet\", \"silent\", \"empty\", or \"sleep\".",
    "Critique this business plan assumption and explain why it's risky: \"We'll capture 10% of the market in year one because our product is better than the competition's.\"",
    "A cyclist rides uphill at 8 mph and returns the same route downhill at 20 mph. If the total round trip takes 3.5 hours, how long is the route one way? Show your reasoning.",
    "Propose a step-by-step plan for transitioning a family business from the founding generation to the next, and justify the ordering of the steps.",
    "Write a short scene, four to six lines of dialogue, between two coworkers disagreeing about a deadline, without either character raising their voice or using an exclamation point.",
    "Evaluate whether a college student should take an unpaid internship in their field or a paid job outside their field for the summer, and recommend one with reasoning.",
    "If three workers can build a fence in 4 days, how many days would it take five workers at the same rate, and what assumption does that calculation rely on? Explain.",
    "Design a prioritized three-step plan for a city considering converting an underused parking lot into a public park, and justify the order.",
    "Write a four-line poem about jealousy without using the words \"jealous\", \"envy\", \"green\", or \"want\".",
    "Brainstorm two unconventional ways a small farm could diversify its income beyond selling produce, and explain the reasoning behind each idea.",
    "A rectangular garden is twice as long as it is wide, and its perimeter is 60 feet. What are its dimensions, and how did you find them? Show your steps.",
    "Recommend whether a musician should sign with a label or stay independent given a $50,000 advance offer with reduced royalties versus full royalties alone, and justify the choice.",
    "Write a short reflective paragraph, four to five sentences, about failing at something, from the perspective of the person who failed, without using the word \"fail\" or \"failure\".",
    "Propose a plan for a nonprofit to diversify its funding sources over two years, moving away from reliance on a single major donor, and justify the sequencing.",
    "If a $1,200 laptop loses 25% of its value each year, in how many years will it be worth less than $300? Show your reasoning and round appropriately.",
    "Design a three-phase plan for onboarding a new engineering hire at a startup with no formal onboarding process yet, and explain your ordering.",
    "Write a six-line poem about waiting for test results, without using the words \"worry\", \"anxious\", \"fear\", or \"hope\".",
    "Evaluate the risk of a small business taking on a large loan to expand during a period of economic uncertainty, and recommend a course of action with reasoning.",
    "Two pumps working together fill a reservoir in 3 hours; alone, the faster pump would take 5 hours. How long would the slower pump take alone? Show your work.",
    "Propose three prioritized steps a remote team could take to rebuild trust after a missed product deadline, and justify the ordering.",
    "Write a short parable, five to six sentences, about patience, without directly using the word \"patience\" or \"patient\".",
    "Critique this decision and explain the flaw in reasoning: \"We should discontinue the product because sales dropped last month, even though it's the first month after a price increase.\"",
    "If a plane flies 500 miles against a headwind in 2.5 hours and the same 500 miles with a tailwind in 2 hours, what is the wind speed? Show your reasoning.",
    "Design a plan for a couple to merge finances after getting married, prioritizing the steps that reduce the most friction first, and justify the order.",
    "Write a four-line poem about a first snowfall without using the words \"snow\", \"white\", \"cold\", or \"winter\".",
    "Recommend whether a restaurant should raise menu prices or shrink portion sizes to offset rising ingredient costs, and justify the choice with the tradeoffs involved.",
    "A worker's hourly wage increases 5% each year. Starting at $20/hour, in what year does their wage first exceed $28/hour? Show the calculation.",
    "Propose a prioritized three-step plan for a small town to reduce traffic congestion downtown without banning cars outright, and justify the ordering.",
    "Write a short piece of flash fiction, five to seven sentences, about a character who finds an old key, without revealing what it unlocks.",
    "Evaluate whether a college should require students to complete an internship to graduate, considering both access and equity concerns, and give a recommendation.",
    "Two candles of equal length burn at different rates; one takes 4 hours to burn out and the other 6 hours. If both are lit at the same time, after how long will one candle be exactly twice the height of the other? Show your reasoning.",
    "Design a three-step plan for a musician to grow an audience on a limited budget, ordered by expected impact, and justify the ordering.",
    "Write a four-line poem about forgiveness without using the words \"forgive\", \"sorry\", \"wrong\", or \"heart\".",
    "Recommend how a manager should handle a high performer who consistently misses team meetings, weighing the tradeoffs, and justify the recommendation.",
]


def build() -> list[dict]:
    """Assemble the labeled rows, deduplicating within each tier."""
    rows = []
    for prefix, tier, prompts in (
        ("t1", TIER_SIMPLE, TIER1_PROMPTS),
        ("t2", TIER_MODERATE, TIER2_PROMPTS),
        ("t3", TIER_COMPLEX, TIER3_PROMPTS),
    ):
        seen = set()
        deduped = [p for p in prompts if not (p in seen or seen.add(p))]
        if len(deduped) != len(prompts):
            print(f"  {prefix}: dropped {len(prompts) - len(deduped)} duplicate prompt(s)")
        for i, prompt in enumerate(deduped, start=1):
            rows.append({"id": f"{prefix}-{i:03d}", "tier": tier, "prompt": prompt})
    return rows


def main() -> int:
    rows = build()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    by_tier: dict[int, int] = {}
    for row in rows:
        by_tier[row["tier"]] = by_tier.get(row["tier"], 0) + 1
    print(f"\nwrote {len(rows)} labeled prompts to {OUT_FILE}")
    for tier, count in sorted(by_tier.items()):
        print(f"  tier {tier}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
