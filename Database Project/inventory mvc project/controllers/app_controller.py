from models.db import Database

class AppController:
    def __init__(self):
        self.db = Database()

    def list_products(self):
        return self.db.execute("SELECT * FROM Product ORDER BY name;", fetch=True)

    def place_order(self, customer_id, product_id, quantity):
        return self.db.call_proc('place_order', [customer_id, product_id, quantity])

    def get_price(self, product_id):
        return self.db.execute("SELECT get_product_price(%s) AS price;", [product_id], fetch=True)

    def customer_orders(self, cust_id):
        return self.db.execute("SELECT order_id, order_date FROM customer_orders WHERE customer_id = %s;", [cust_id], fetch=True)

    def complex_report(self):
        return self.db.execute('''SELECT p.name, SUM(oi.quantity) AS total_qty FROM Product p JOIN OrderItem oi ON p.product_id = oi.product_id GROUP BY p.name HAVING total_qty > 0 ORDER BY total_qty DESC;''', fetch=True)

    def close(self):
        self.db.close()