# creates updated feature tables
from config import market_data_clickhouse_ip
import clickhouse_connect

def market_clickhouse_client():      
    client = clickhouse_connect.get_client(
        host=market_data_clickhouse_ip,
        port=8123,
        username="default",   
        password="mysecurepassword",
        database="default"
    )
    return client

def get_ochlv(client):
    ochlv_df = client.query_df("""
        SELECT avg(price) AS price
        FROM ticks_db
        WHERE timestamp > now() - INTERVAL 2 HOUR
        GROUP BY toStartOfMinute(timestamp)
        ORDER BY toStartOfMinute(timestamp) DESC
        LIMIT 120                                
    """)
    return ochlv_df


