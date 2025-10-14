import requests

class ZendeskClient:
    def __init__(self, subdomain, email, token):
        self.url = f"https://{subdomain}.zendesk.com/api/v2"
        self.auth = (email + "/token", token)

    def create_ticket(self, subject, description, requester_email):
        data = {
            "ticket": {
                "subject": subject,
                "description": description,
                "requester": {"email": requester_email}
            }
        }
        response = requests.post(f"{self.url}/tickets.json", json=data, auth=self.auth)
        return response.json()
