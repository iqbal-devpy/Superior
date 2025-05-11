class AppView:
    @staticmethod
    def show_menu():
        print("--- Inventory Management (MySQL) ---")
        print("1. List Products")
        print("2. Place Order")
        print("3. Get Product Price")
        print("4. Customer Orders")
        print("5. Sales Report")
        print("0. Exit")

    @staticmethod
    def get_input(prompt):
        return input(f"{prompt}: ")

    @staticmethod
    def display(records):
        for r in records:
            print(r)