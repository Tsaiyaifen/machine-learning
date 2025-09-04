from db.connection import get_connection

def setup():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        date DATE,
        product VARCHAR(50),
        quantity INT,
        price FLOAT
    )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    setup()
