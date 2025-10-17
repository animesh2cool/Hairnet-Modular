"""
Database Connection and Operations
Handles MySQL database interactions
"""
import pymysql
import logging
from typing import Optional
from .config import DatabaseConfig

logger = logging.getLogger(__name__)


def create_db_connection() -> Optional[pymysql.Connection]:
    """
    Create database connection
    Returns: MySQL connection object or None
    """
    try:
        connection = pymysql.connect(**DatabaseConfig.get_config_dict())
        return connection
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None


def create_violations_table() -> bool:
    """
    Create violations table if not exists
    Returns: True if successful, False otherwise
    """
    connection = create_db_connection()
    if not connection:
        return False
    
    try:
        with connection.cursor() as cursor:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS violations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(20) NOT NULL,
                employee_id VARCHAR(50) DEFAULT 'Unknown',
                employee_name VARCHAR(100) DEFAULT 'Unknown',
                confidence DECIMAL(4,2) NOT NULL,
                face_confidence DECIMAL(6,3) NOT NULL,
                violation_image TEXT,
                violation_faces TEXT,
                violation_video TEXT,
                original_known_faces TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_employee_id (employee_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            cursor.execute(create_table_sql)
            connection.commit()
            logger.info("✅ Violations table created/verified successfully")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to create violations table: {e}")
        return False
    finally:
        connection.close()


def save_violation_to_database(
    timestamp: str,
    employee_id: Optional[str],
    employee_name: Optional[str],
    confidence: float,
    face_confidence: float,
    violation_image: str,
    violation_faces: str,
    violation_video: str,
    original_known_faces: str
) -> bool:
    """
    Save violation record to database
    Returns: True if successful, False otherwise
    """
    connection = create_db_connection()
    if not connection:
        return False
    
    try:
        with connection.cursor() as cursor:
            insert_sql = """
            INSERT INTO violations 
            (timestamp, employee_id, employee_name, confidence, face_confidence, 
             violation_image, violation_faces, violation_video, original_known_faces)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, (
                timestamp,
                employee_id or 'Unknown',
                employee_name or 'Unknown',
                float(confidence),
                float(face_confidence),
                violation_image or '',
                violation_faces or '',
                violation_video or '',
                original_known_faces or ''
            ))
            connection.commit()
            logger.info(f"✅ Violation saved to database: {employee_name} ({employee_id})")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to save violation to database: {e}")
        return False
    finally:
        connection.close()


def get_violations_from_database(limit: int = 50, offset: int = 0) -> dict:
    """
    Get violations from database with pagination
    Returns: Dictionary with violations and metadata
    """
    connection = create_db_connection()
    if not connection:
        return {"error": "Database connection failed", "violations": [], "total": 0}
    
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # Get total count
            cursor.execute("SELECT COUNT(*) as total FROM violations")
            total = cursor.fetchone()['total']
            
            # Get violations with pagination
            cursor.execute("""
                SELECT id, timestamp, employee_id, employee_name, confidence, 
                       face_confidence, violation_image, violation_faces, 
                       violation_video, original_known_faces, created_at
                FROM violations 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            violations = cursor.fetchall()
            
            return {
                "violations": violations,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
    except Exception as e:
        logger.error(f"Error fetching violations from database: {e}")
        return {"error": str(e), "violations": [], "total": 0}
    finally:
        connection.close()


def check_database_health() -> dict:
    """
    Check database connection and table status
    Returns: Health status dictionary
    """
    connection = create_db_connection()
    if not connection:
        return {"status": "error", "message": "Cannot connect to database"}
    
    try:
        with connection.cursor() as cursor:
            # Check if violations table exists
            cursor.execute("SHOW TABLES LIKE 'violations'")
            table_exists = cursor.fetchone() is not None
            
            # Get violations count
            violations_count = 0
            if table_exists:
                cursor.execute("SELECT COUNT(*) as count FROM violations")
                violations_count = cursor.fetchone()[0]
            
            return {
                "status": "healthy",
                "database_connected": True,
                "violations_table_exists": table_exists,
                "total_violations_in_db": violations_count,
                "database_config": {
                    "host": DatabaseConfig.HOST,
                    "port": DatabaseConfig.PORT,
                    "database": DatabaseConfig.DATABASE
                }
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        connection.close()