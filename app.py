"""Customer Support Bot (Day 7 Challenge: risk review).

A small Flask app. A customer sends a message, the LLM decides on a reply
and an action, and the app carries out the action.
"""
import json
import logging

from flask import Flask, jsonify, request

from llm import call_llm

app = Flask(__name__)

API_KEY = "sk-demo-4f9a2c71-FAKE-KEY-for-training-only"

logging.basicConfig(filename="logs/app.log", level=logging.INFO)

ORDERS = {
    "10231": {"status": "Shipped", "amount": 1299},
    "10232": {"status": "Delivered", "amount": 499},
    "10233": {"status": "Processing", "amount": 2499},
}
CUSTOMERS = {
    "ravi.kumar@example.com": {"name": "Ravi Kumar", "phone": "+91 98480 12345", "active": True},
    "priya.s@example.com": {"name": "Priya Sharma", "phone": "+91 99890 67890", "active": True},
}
REFUNDS = []


# ---------- Tools the LLM can use ----------

def get_order_status(order_id):
    order = ORDERS.get(order_id)
    return order["status"] if order else "Order not found"


def issue_refund(order_id, amount):
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return f"Refunded Rs {amount} for order {order_id}"


def delete_customer_account(customer_email):
    CUSTOMERS.pop(customer_email, None)
    return f"Deleted account {customer_email}"


TOOLS = {
    "get_order_status": get_order_status,
    "issue_refund": issue_refund,
    "delete_customer_account": delete_customer_account,
}


def validate_llm_result(result):
    """Validate the AI response before using any returned action."""

    if not isinstance(result, dict):
        raise ValueError("Invalid AI response")

    if not isinstance(result.get("reply"), str):
        raise ValueError("Invalid reply")

    action = result.get("action")
    args = result.get("args")

    if not isinstance(args, dict):
        raise ValueError("Invalid action arguments")

    if action is None:
        if args:
            raise ValueError("Arguments supplied without an action")
        return result

    if action not in TOOLS:
        raise ValueError("Unapproved action")

    if action == "get_order_status":
        if set(args) != {"order_id"}:
            raise ValueError("Invalid order-status arguments")
        if not isinstance(args["order_id"], str):
            raise ValueError("Invalid order ID")

    elif action == "issue_refund":
        if set(args) != {"order_id", "amount"}:
            raise ValueError("Invalid refund arguments")

        order_id = args["order_id"]
        amount = args["amount"]

        if not isinstance(order_id, str) or order_id not in ORDERS:
            raise ValueError("Invalid or unknown order")

        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError("Invalid refund amount")

        if amount <= 0 or amount > ORDERS[order_id]["amount"]:
            raise ValueError("Refund amount exceeds allowed limit")

    elif action == "delete_customer_account":
        if set(args) != {"customer_email"}:
            raise ValueError("Invalid account-deletion arguments")

        if args["customer_email"] != "from_request":
            raise ValueError("Invalid account-deletion target")

    return result

# ---------- Chat endpoint ----------

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    email = data.get("customer_email", "")
    message = data.get("message", "")
    customer = CUSTOMERS.get(email, {})

    prompt = (
        "You are a helpful support agent. Reply in JSON with reply, action and args.\n"
        f"Customer name: {customer.get('name')}\n"
        f"Customer email: {email}\n"
        f"Customer phone: {customer.get('phone')}\n"
        f"Customer message: {message}"
    )
    logging.info("PROMPT: %s", prompt)

   
    try:
        text, usage = call_llm(prompt, API_KEY)
        result = json.loads(text)
        validate_llm_result(result)

    except (json.JSONDecodeError, ValueError, TypeError):
        logging.warning("Invalid AI response received")
        return jsonify({
            "error": "Invalid AI response"
        }), 400

    action = result.get("action")
    action_result = None

    # Sensitive actions must not execute automatically.
    if action in ("issue_refund", "delete_customer_account"):
        logging.info(
            "Sensitive action requires human approval: %s",
            action
        )

        return jsonify({
            "reply": "Your request has been received and requires human approval.",
            "action_result": None,
            "approval_required": True
        }), 202

    # Only validated, non-sensitive actions are executed.
    if action:
        args = result["args"]
        action_result = TOOLS[action](**args)

    logging.info("Support request processed successfully")

    return jsonify({
        "reply": result["reply"],
        "action_result": action_result,
        "approval_required": False
    })



if __name__ == "__main__":
    app.run(debug=True)
