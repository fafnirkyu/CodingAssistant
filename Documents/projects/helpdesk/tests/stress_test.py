import requests
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from collections import Counter

API_URL = "http://127.0.0.1:8000/tickets"
POLL_URL = "http://127.0.0.1:8000/ticket/{}"

# Realistic support tickets for Zendesk integration
TICKETS = [
    ("I can't log into my account after resetting my password.", "ACCOUNT"),
    ("My order #12345 hasn't arrived after 2 weeks.", "ORDER"),
    ("You charged me twice for my last purchase on March 15th.", "BILLING"),
    ("I want to cancel my premium subscription immediately.", "SUBSCRIPTION"),
    ("The mobile app keeps crashing when I open it on iOS.", "TECHNICAL"),
    ("I received a damaged product in my order from yesterday.", "ORDER"),
    ("My payment was declined but my card has sufficient funds.", "BILLING"),
    ("The website loads very slowly on my Chrome browser.", "TECHNICAL"),
    ("I updated my email but verification isn't working.", "ACCOUNT"),
    ("How can I change my shipping address for order #67890?", "ORDER"),
    ("I got 'invalid promo code' error with code SAVE20.", "BILLING"),
    ("The live chat feature isn't loading on your website.", "TECHNICAL"),
    ("When does my annual subscription automatically renew?", "SUBSCRIPTION"),
    ("Can I get an invoice for my purchase last week?", "BILLING"),
    ("Tracking shows delivered but I never received package.", "ORDER"),
    ("I forgot my username and recovery email isn't working.", "ACCOUNT"),
    ("The refund for my cancelled order hasn't processed.", "BILLING"),
    ("I can't add my credit card - says invalid card number.", "BILLING"),
    ("My account was locked after multiple failed login attempts.", "ACCOUNT"),
    ("Order confirmation emails are not coming through.", "ORDER")
]

# Send a ticket and poll until analyzed or timeout
def send_ticket_task(message, expected, max_wait_time=60, simulate_think=True):
    session = requests.Session()
    request_id = random.randint(1000, 9999)
    # Simulate user think time inside the task so submission is concurrent
    if simulate_think:
        time.sleep(random.uniform(0.5, 3.5))

    start = time.time()
    try:
        print(f"🚀 [{request_id}] SENDING: '{message[:60]}...'")
        resp = session.post(API_URL, json={"message": message}, timeout=30)
        if resp.status_code not in (200, 201):
            elapsed = round(time.time() - start, 2)
            print(f"❌ [{request_id}] Create failed {resp.status_code} in {elapsed}s")
            return {
                "message": message,
                "status": resp.status_code,
                "elapsed": elapsed,
                "category": "?",
                "expected": expected,
                "correct": False
            }
        ticket = resp.json()
        ticket_id = ticket.get("id")
        if not ticket_id:
            elapsed = round(time.time() - start, 2)
            print(f"❌ [{request_id}] No ticket id returned in {elapsed}s")
            return {
                "message": message,
                "status": "NO_ID",
                "elapsed": elapsed,
                "category": "?",
                "expected": expected,
                "correct": False
            }
        print(f"📝 [{request_id}] Ticket {ticket_id} created")
        # Polling with exponential backoff (but keep responsive)
        poll_delay = 1.0
        elapsed = None
        while (time.time() - start) < max_wait_time:
            time.sleep(poll_delay)
            try:
                poll = session.get(POLL_URL.format(ticket_id), timeout=10)
                if poll.status_code == 200:
                    data = poll.json()
                    if data.get("analyzed", False):
                        elapsed = round(time.time() - start, 2)
                        category = data.get("category", "?")
                        print(f"[{request_id}] {category} in {elapsed}s (ticket {ticket_id})")
                        return {
                            "message": message,
                            "status": 200,
                            "elapsed": elapsed,
                            "category": category,
                            "summary": data.get("summary", "?"),
                            "ticket_id": ticket_id,
                            "expected": expected,
                            "correct": category.upper() == expected.upper()
                        }
                # backoff strategy, but cap at 5s
                poll_delay = min(5.0, poll_delay * 1.5)
            except Exception as e:
                # transient network error - continue until max_wait_time
                print(f"⚠️ [{request_id}] Poll error: {e}")
                poll_delay = min(5.0, poll_delay * 1.5)
        # timed out
        elapsed = round(time.time() - start, 2)
        print(f"[{request_id}] Timeout after {elapsed}s (ticket {ticket_id})")
        return {
            "message": message,
            "status": "TIMEOUT",
            "elapsed": elapsed,
            "category": "?",
            "ticket_id": ticket_id,
            "expected": expected,
            "correct": False
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"[{request_id}] Exception: {e}")
        return {
            "message": message,
            "status": "ERROR",
            "elapsed": elapsed,
            "category": "?",
            "expected": expected,
            "correct": False
        }
    finally:
        session.close()

# Main stress test function
def run_zendesk_stress_test(concurrent_users=10, tickets_per_user=5):
    total = concurrent_users * tickets_per_user
    print("ZENDESK-LIKE STRESS TEST")
    print(f"   Concurrent Workers: {concurrent_users}")
    print(f"   Tickets per Worker: {tickets_per_user}")
    print(f"   Total Tickets: {total}")
    print("=" * 60)
    
    # Build task list (message, expected)
    test_tickets = []
    # If we don't have enough distinct tickets, reuse them randomly
    for i in range(total):
        msg, expected = TICKETS[i % len(TICKETS)]
        test_tickets.append((msg, expected))
    
    results = []
    # Use ThreadPoolExecutor with connection pooling in each task
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = {
            executor.submit(send_ticket_task, msg, expected, 60, simulate_think=True): (msg, expected)
            for (msg, expected) in test_tickets
        }
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            status_icon = "✅" if res['status'] == 200 else "❌"
            print(f"{status_icon} [{completed}/{len(test_tickets)}] {res.get('category')} - {res.get('elapsed')}s (expected: {res.get('expected')})")
    
    # Summarize metrics
    successful = [r for r in results if r['status'] == 200]
    if successful:
        avg_time = mean([r['elapsed'] for r in successful])
    else:
        avg_time = None

    correct_count = sum(1 for r in successful if r.get("correct"))
    success_rate = len(successful) / len(test_tickets) * 100
    accuracy = (correct_count / len(successful) * 100) if successful else 0.0
    
    print("\n" + "="*60)
    print("ZENDESK INTEGRATION READINESS RESULTS")
    print("="*60)
    print(f"Success Rate: {len(successful)}/{len(test_tickets)} ({success_rate:.1f}%)")
    print(f"Classification Accuracy (on successful responses): {accuracy:.1f}%")
    if avg_time is not None:
        print(f"⏱️  Average Response Time (successful): {avg_time:.2f}s")
    
    # Performance tiers
    excellent = len([r for r in successful if r['elapsed'] < 10])
    good = len([r for r in successful if 10 <= r['elapsed'] < 20])
    poor = len([r for r in successful if r['elapsed'] >= 20])
    print(f"\nPerformance Tiers:")
    print(f"   Excellent (<10s): {excellent} tickets")
    print(f"   Good (10-20s): {good} tickets")
    print(f"   Poor (≥20s): {poor} tickets")
    
    # Category distribution
    category_dist = Counter([r['category'] for r in successful])
    print(f"\nCategory Distribution:")
    for category, count in category_dist.most_common():
        print(f"   {category}: {count}")
    
    # Save results
    with open("zendesk_stress_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to zendesk_stress_results.json")
    
    # Simple readiness heuristic
    if (len(successful) >= len(test_tickets) * 0.8) and (accuracy >= 80):
        print("\nREADY FOR ZENDESK INTEGRATION!")
    else:
        print("\nNEEDS IMPROVEMENT BEFORE ZENDESK INTEGRATION")
    
    return results

if __name__ == "__main__":
    # Example run: 5 concurrent workers, 4 tickets each = 20 requests
    run_zendesk_stress_test(concurrent_users=5, tickets_per_user=4)
