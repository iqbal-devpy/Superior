CREATE DATABASE IF NOT EXISTS inventory_db;
USE inventory_db;

-- DROP existing tables
DROP TABLE IF EXISTS OrderItem;
DROP TABLE IF EXISTS `Order`;
DROP TABLE IF EXISTS Customer;
DROP TABLE IF EXISTS Product;

-- CREATE tables (DDL)
CREATE TABLE Product (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    stock INT DEFAULT 0,
    price DECIMAL(10,2) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE Customer (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL
) ENGINE=InnoDB;

CREATE TABLE `Order` (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES Customer(customer_id)
) ENGINE=InnoDB;

CREATE TABLE OrderItem (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES `Order`(order_id),
    FOREIGN KEY (product_id) REFERENCES Product(product_id)
) ENGINE=InnoDB;

-- DML: sample data
INSERT INTO Product (name, stock, price) VALUES
('Chai Cup', 200, 3.50),
('Desk Fan', 80, 45.00),
('Laptop', 25, 1200.00),
('Notebook', 500, 1.25),
('Smartphone Charger', 150, 15.99),
('LED Bulb', 100, 1.99),
('Keyboard', 70, 25.00);

INSERT INTO Customer (name, email) VALUES
('Ali Khan', 'ali.khan@pakmail.com'),
('Fatima Baloch', 'fatima.baloch@pakmail.com'),
('Ahmed Nawaz', 'ahmed.nawaz@pakmail.com'),
('Ayesha Noor', 'ayesha.noor@pakmail.com'),
('Zain Ahmed', 'zain.ahmed@pakmail.com'),
('Usman Tariq', 'usman.tariq@pakmail.com');

-- UPDATE & DELETE examples
UPDATE Product SET stock = stock - 10 WHERE product_id = 1;
DELETE FROM Customer WHERE name = 'Bob';

-- Complex SELECT with JOINs, WHERE, GROUP BY, HAVING, ORDER BY
SELECT p.name,
       SUM(oi.quantity) AS total_qty
FROM Product p
JOIN OrderItem oi ON p.product_id = oi.product_id
GROUP BY p.name
HAVING SUM(oi.quantity) > 5
ORDER BY total_qty DESC;

-- Stored Procedure: place_order
DELIMITER $$
CREATE PROCEDURE place_order(
    IN p_customer INT,
    IN p_product INT,
    IN p_qty INT
)
BEGIN
    DECLARE temp_order_id INT;
    START TRANSACTION;
    IF (SELECT stock FROM Product WHERE product_id = p_product) < p_qty THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Insufficient stock';
    END IF;
    INSERT INTO `Order` (customer_id) VALUES (p_customer);
    SET temp_order_id = LAST_INSERT_ID();
    UPDATE Product SET stock = stock - p_qty WHERE product_id = p_product;
    INSERT INTO OrderItem (order_id, product_id, quantity)
      VALUES (temp_order_id, p_product, p_qty);
    COMMIT;
END$$
DELIMITER ;

-- Scalar Function: get_product_price
DELIMITER $$
CREATE FUNCTION get_product_price(p_id INT)
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
    DECLARE p DECIMAL(10,2);
    SELECT price INTO p FROM Product WHERE product_id = p_id;
    RETURN p;
END$$
DELIMITER ;

-- View for inline table-valued function equivalent
CREATE OR REPLACE VIEW customer_orders AS
SELECT o.order_id, o.order_date, o.customer_id
FROM `Order` o;

-- BEFORE trigger
DELIMITER $$
CREATE TRIGGER trg_before_order
BEFORE INSERT ON `Order`
FOR EACH ROW
BEGIN
    IF NEW.order_date IS NULL THEN
        SET NEW.order_date = CURRENT_TIMESTAMP;
    END IF;
END$$
DELIMITER ;

-- AFTER trigger
DELIMITER $$
CREATE TRIGGER trg_after_order
AFTER INSERT ON `Order`
FOR EACH ROW
BEGIN
    INSERT INTO OrderItem (order_id, product_id, quantity)
    SELECT NEW.order_id, product_id, quantity
    FROM OrderItem WHERE order_id = NEW.order_id;
END$$
DELIMITER ;

-- Transaction management example
-- START TRANSACTION;
-- UPDATE Product SET stock = stock - 5 WHERE product_id = 2;
-- DELETE FROM OrderItem WHERE order_item_id = 99;
-- ROLLBACK;