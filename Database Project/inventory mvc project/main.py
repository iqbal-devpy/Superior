from controllers.app_controller import AppController # type: ignore
from views.app_view import AppView # type: ignore

if __name__ == '__main__':
    ctrl = AppController()
    view = AppView()

    while True:
        view.show_menu()
        choice = view.get_input('Select')
        if choice == '1':
            products = ctrl.list_products()
            view.display(products)
        elif choice == '2':
            cid = int(view.get_input('Customer ID'))
            pid = int(view.get_input('Product ID'))
            qty = int(view.get_input('Quantity'))
            try:
                ctrl.place_order(cid, pid, qty)
                print("Order placed successfully.")
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '3':
            pid = int(view.get_input('Product ID'))
            price = ctrl.get_price(pid)
            print(f"Price: {price[0]['price']}")
        elif choice == '4':
            cid = int(view.get_input('Customer ID'))
            orders = ctrl.customer_orders(cid)
            view.display(orders)
        elif choice == '5':
            report = ctrl.complex_report()
            view.display(report)
        elif choice == '0':
            break
        else:
            print("Invalid choice.")

    ctrl.close()