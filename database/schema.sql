-- User Data Persistence Schema
-- This schema stores all user investment data persistently

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(36),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Investment sessions table
CREATE TABLE IF NOT EXISTS investment_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    initial_investment DECIMAL(15,2) NOT NULL,
    current_value DECIMAL(15,2) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('idle', 'active', 'paused')),
    started_at TIMESTAMP NOT NULL,
    paused_at TIMESTAMP NULL,
    ended_at TIMESTAMP NULL,
    total_pnl DECIMAL(15,2) DEFAULT 0.0,
    total_orders INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- PnL records table
CREATE TABLE IF NOT EXISTS pnl_records (
    record_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    portfolio_value DECIMAL(15,2) NOT NULL,
    pnl_amount DECIMAL(15,2) NOT NULL,
    pnl_percentage DECIMAL(8,4) NOT NULL,
    market_change DECIMAL(8,4) NOT NULL,
    order_pnl_impact DECIMAL(8,4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES investment_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Trading orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    order_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('hold', 'buy', 'sell')),
    confidence DECIMAL(5,4) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'executed', 'rejected', 'expired')),
    pnl_impact DECIMAL(8,4) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    features_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES investment_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction INTEGER NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    features_hash VARCHAR(64) NOT NULL,
    execution_time_ms DECIMAL(8,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES investment_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- User sessions table (for authentication)
CREATE TABLE IF NOT EXISTS user_sessions (
    session_token VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_investment_sessions_user_id ON investment_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_investment_sessions_status ON investment_sessions(status);
CREATE INDEX IF NOT EXISTS idx_pnl_records_session_id ON pnl_records(session_id);
CREATE INDEX IF NOT EXISTS idx_trading_orders_session_id ON trading_orders(session_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_session_id ON model_predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Views for common queries
CREATE OR REPLACE VIEW user_portfolio_summary AS
SELECT 
    u.user_id,
    u.username,
    COUNT(DISTINCT inv_sess.session_id) as total_sessions,
    SUM(CASE WHEN inv_sess.status = 'active' THEN 1 ELSE 0 END) as active_sessions,
    SUM(inv_sess.total_pnl) as total_lifetime_pnl,
    AVG(inv_sess.total_pnl) as avg_session_pnl,
    MAX(inv_sess.current_value) as highest_portfolio_value
FROM users u
LEFT JOIN investment_sessions inv_sess ON u.user_id = inv_sess.user_id
GROUP BY u.user_id, u.username;

CREATE OR REPLACE VIEW session_performance AS
SELECT 
    inv_sess.session_id,
    inv_sess.user_id,
    u.username,
    inv_sess.initial_investment,
    inv_sess.current_value,
    inv_sess.total_pnl,
    inv_sess.total_orders,
    inv_sess.started_at,
    inv_sess.paused_at,
    inv_sess.ended_at,
    ROUND(((inv_sess.current_value - inv_sess.initial_investment) / inv_sess.current_value) * 100, 2) as pnl_percentage
FROM investment_sessions inv_sess
JOIN users u ON inv_sess.user_id = u.user_id
ORDER BY inv_sess.started_at DESC;
