#!/usr/bin/env python3
"""
Model Adapters & Deployment Engine
Manages model versioning, persistence, and deployment pipeline
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psycopg2
import redis
import requests
import traceback
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ModelMetadata:
    """Model metadata for versioning and tracking"""
    model_id: str
    model_name: str
    model_type: str  # 'lightgbm', 'extreme_trees', 'ensemble'
    version: str
    created_at: str
    trained_at: str
    performance_score: float
    evaluation_metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    model_size_bytes: int
    status: str  # 'trained', 'evaluated', 'deployed', 'archived'
    deployment_date: Optional[str] = None
    deployment_performance: Optional[Dict[str, Any]] = None

class ModelAdapter:
    """Base class for model adapters"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.metadata = None
    
    def save_model(self, model, metadata: ModelMetadata, save_path: str) -> bool:
        """Save model with metadata"""
        try:
            # Save the model
            if self.model_type == 'lightgbm':
                model.save_model(f"{save_path}.txt")
            else:
                joblib.dump(model, f"{save_path}.joblib")
            
            # Save metadata
            with open(f"{save_path}_metadata.json", 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, load_path: str):
        """Load model from path"""
        try:
            if self.model_type == 'lightgbm':
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=f"{load_path}.txt")
            else:
                self.model = joblib.load(f"{load_path}.joblib")
            
            # Load metadata
            with open(f"{load_path}_metadata.json", 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = ModelMetadata(**metadata_dict)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            if self.model_type == 'lightgbm':
                return self.model.predict(X)
            else:
                return self.model.predict(X)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return np.array([])

class ModelManager:
    """Manages model lifecycle, versioning, and deployment"""
    
    def __init__(self, postgres_url: str, redis_url: str):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.redis_client = None
        self.postgres_conn = None
        self.init_connections()
        self.init_database()
    
    def init_connections(self):
        """Initialize database connections"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            
            # PostgreSQL connection
            self.postgres_conn = psycopg2.connect(self.postgres_url)
            self.postgres_conn.autocommit = True
            
        except Exception as e:
            print(f"Error initializing connections: {e}")
    
    def init_database(self):
        """Initialize database tables for model management"""
        try:
            cursor = self.postgres_conn.cursor()
            
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) UNIQUE NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    trained_at TIMESTAMP NOT NULL,
                    performance_score DECIMAL(10,6),
                    evaluation_metrics JSONB,
                    hyperparameters JSONB,
                    feature_names TEXT[],
                    model_size_bytes BIGINT,
                    status VARCHAR(50) NOT NULL,
                    deployment_date TIMESTAMP,
                    deployment_performance JSONB,
                    model_path VARCHAR(500),
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create model_versions table for versioning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    model_path VARCHAR(500) NOT NULL,
                    metadata_path VARCHAR(500) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_id, version)
                )
            """)
            
            # Create deployments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(255) NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_by VARCHAR(255),
                    deployment_notes TEXT,
                    performance_metrics JSONB,
                    status VARCHAR(50) DEFAULT 'active'
                )
            """)
            
            cursor.close()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def register_model(self, model, metadata: ModelMetadata, model_path: str) -> bool:
        """Register a new model in the system"""
        try:
            cursor = self.postgres_conn.cursor()
            
            # Insert into models table
            cursor.execute("""
                INSERT INTO models (
                    model_id, model_name, model_type, version, created_at, trained_at,
                    performance_score, evaluation_metrics, hyperparameters, feature_names,
                    model_size_bytes, status, model_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    version = EXCLUDED.version,
                    performance_score = EXCLUDED.performance_score,
                    evaluation_metrics = EXCLUDED.evaluation_metrics,
                    status = EXCLUDED.status,
                    model_path = EXCLUDED.model_path
            """, (
                metadata.model_id, metadata.model_name, metadata.model_type,
                metadata.version, metadata.created_at, metadata.trained_at,
                metadata.performance_score, json.dumps(metadata.evaluation_metrics),
                json.dumps(metadata.hyperparameters), metadata.feature_names,
                metadata.model_size_bytes, metadata.status, model_path
            ))
            
            # Insert into model_versions table
            cursor.execute("""
                INSERT INTO model_versions (model_id, version, model_path, metadata_path)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (model_id, version) DO UPDATE SET
                    model_path = EXCLUDED.model_path,
                    metadata_path = EXCLUDED.metadata_path
            """, (
                metadata.model_id, metadata.version, model_path,
                f"{model_path}_metadata.json"
            ))
            
            cursor.close()
            
            # Cache model metadata in Redis
            self.redis_client.setex(
                f"model:{metadata.model_id}:{metadata.version}",
                3600,  # 1 hour TTL
                json.dumps(asdict(metadata))
            )
            
            return True
            
        except Exception as e:
            print(f"Error registering model: {e}")
            traceback.print_exc()
            return False
    
    def get_model_metadata(self, model_id: str, version: str = None) -> Optional[ModelMetadata]:
        """Get model metadata"""
        try:
            # Try Redis first
            if version:
                cache_key = f"model:{model_id}:{version}"
            else:
                cache_key = f"model:{model_id}:latest"
            
            cached = self.redis_client.get(cache_key)
            if cached:
                metadata_dict = json.loads(cached)
                return ModelMetadata(**metadata_dict)
            
            # Fallback to PostgreSQL
            cursor = self.postgres_conn.cursor()
            
            if version:
                cursor.execute("""
                    SELECT * FROM models WHERE model_id = %s AND version = %s
                """, (model_id, version))
            else:
                cursor.execute("""
                    SELECT * FROM models WHERE model_id = %s 
                    ORDER BY created_at DESC LIMIT 1
                """, (model_id,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                metadata_dict = {
                    'model_id': row[1],
                    'model_name': row[2],
                    'model_type': row[3],
                    'version': row[4],
                    'created_at': row[5].isoformat(),
                    'trained_at': row[6].isoformat(),
                    'performance_score': float(row[7]) if row[7] else 0.0,
                    'evaluation_metrics': row[8] if row[8] else {},
                    'hyperparameters': row[9] if row[9] else {},
                    'feature_names': row[10] if row[10] else [],
                    'model_size_bytes': row[11] if row[11] else 0,
                    'status': row[12],
                    'deployment_date': row[13].isoformat() if row[13] else None,
                    'deployment_performance': row[14] if row[14] else {}
                }
                
                metadata = ModelMetadata(**metadata_dict)
                
                # Cache in Redis
                self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(asdict(metadata))
                )
                
                return metadata
            
            return None
            
        except Exception as e:
            print(f"Error getting model metadata: {e}")
            return None
    
    def list_models(self, model_type: str = None, status: str = None) -> List[ModelMetadata]:
        """List all models with optional filtering"""
        try:
            cursor = self.postgres_conn.cursor()
            
            query = "SELECT * FROM models WHERE 1=1"
            params = []
            
            if model_type:
                query += " AND model_type = %s"
                params.append(model_type)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
            
            models = []
            for row in rows:
                metadata_dict = {
                    'model_id': row[1],
                    'model_name': row[2],
                    'model_type': row[3],
                    'version': row[4],
                    'created_at': row[5].isoformat(),
                    'trained_at': row[6].isoformat(),
                    'performance_score': float(row[7]) if row[7] else 0.0,
                    'evaluation_metrics': row[8] if row[8] else {},
                    'hyperparameters': row[9] if row[9] else {},
                    'feature_names': row[10] if row[10] else [],
                    'model_size_bytes': row[11] if row[11] else 0,
                    'status': row[12],
                    'deployment_date': row[13].isoformat() if row[13] else None,
                    'deployment_performance': row[14] if row[14] else {}
                }
                
                models.append(ModelMetadata(**metadata_dict))
            
            return models
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def deploy_model(self, model_id: str, version: str, deployed_by: str = "system") -> bool:
        """Deploy a model to production"""
        try:
            cursor = self.postgres_conn.cursor()
            
            # Update model status to deployed
            cursor.execute("""
                UPDATE models 
                SET status = 'deployed', deployment_date = CURRENT_TIMESTAMP
                WHERE model_id = %s AND version = %s
            """, (model_id, version))
            
            # Archive other versions of the same model
            cursor.execute("""
                UPDATE models 
                SET status = 'archived'
                WHERE model_id = %s AND version != %s AND status = 'deployed'
            """, (model_id, version))
            
            # Insert deployment record
            cursor.execute("""
                INSERT INTO deployments (model_id, version, deployed_by)
                VALUES (%s, %s, %s)
            """, (model_id, version, deployed_by))
            
            cursor.close()
            
            # Update Redis cache
            cache_key = f"model:{model_id}:{version}"
            cached = self.redis_client.get(cache_key)
            if cached:
                metadata_dict = json.loads(cached)
                metadata_dict['status'] = 'deployed'
                metadata_dict['deployment_date'] = datetime.now().isoformat()
                self.redis_client.setex(cache_key, 3600, json.dumps(metadata_dict))
            
            return True
            
        except Exception as e:
            print(f"Error deploying model: {e}")
            return False
    
    def archive_model(self, model_id: str, version: str) -> bool:
        """Archive a model"""
        try:
            cursor = self.postgres_conn.cursor()
            
            cursor.execute("""
                UPDATE models 
                SET status = 'archived'
                WHERE model_id = %s AND version = %s
            """, (model_id, version))
            
            cursor.close()
            
            # Update Redis cache
            cache_key = f"model:{model_id}:{version}"
            cached = self.redis_client.get(cache_key)
            if cached:
                metadata_dict = json.loads(cached)
                metadata_dict['status'] = 'archived'
                self.redis_client.setex(cache_key, 3600, json.dumps(metadata_dict))
            
            return True
            
        except Exception as e:
            print(f"Error archiving model: {e}")
            return False
    
    def get_deployment_history(self, model_id: str = None) -> List[Dict]:
        """Get deployment history"""
        try:
            cursor = self.postgres_conn.cursor()
            
            if model_id:
                cursor.execute("""
                    SELECT * FROM deployments WHERE model_id = %s ORDER BY deployed_at DESC
                """, (model_id,))
            else:
                cursor.execute("""
                    SELECT * FROM deployments ORDER BY deployed_at DESC
                """)
            
            rows = cursor.fetchall()
            cursor.close()
            
            deployments = []
            for row in rows:
                deployments.append({
                    'id': row[0],
                    'model_id': row[1],
                    'version': row[2],
                    'deployed_at': row[3].isoformat() if row[3] else None,
                    'deployed_by': row[4],
                    'deployment_notes': row[5],
                    'performance_metrics': row[6] if row[6] else {},
                    'status': row[7]
                })
            
            return deployments
            
        except Exception as e:
            print(f"Error getting deployment history: {e}")
            return []
    
    def validate_model_for_deployment(self, model_id: str, version: str) -> Tuple[bool, str]:
        """Validate if a model is ready for deployment"""
        try:
            metadata = self.get_model_metadata(model_id, version)
            if not metadata:
                return False, "Model not found"
            
            if metadata.status != 'evaluated':
                return False, f"Model status is {metadata.status}, must be 'evaluated'"
            
            if metadata.performance_score < 0.05:  # Minimum 5% PnL
                return False, f"Performance score {metadata.performance_score} below threshold"
            
            if not metadata.evaluation_metrics:
                return False, "No evaluation metrics available"
            
            return True, "Model ready for deployment"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances"""
        try:
            models = self.list_models()
            
            summary = {
                'total_models': len(models),
                'by_type': {},
                'by_status': {},
                'performance_stats': {
                    'best_score': 0.0,
                    'avg_score': 0.0,
                    'total_deployed': 0
                }
            }
            
            if models:
                scores = [m.performance_score for m in models if m.performance_score]
                summary['performance_stats']['best_score'] = max(scores) if scores else 0.0
                summary['performance_stats']['avg_score'] = np.mean(scores) if scores else 0.0
                summary['performance_stats']['total_deployed'] = len([m for m in models if m.status == 'deployed'])
                
                # Count by type
                for model in models:
                    model_type = model.model_type
                    if model_type not in summary['by_type']:
                        summary['by_type'][model_type] = 0
                    summary['by_type'][model_type] += 1
                
                # Count by status
                for model in models:
                    status = model.status
                    if status not in summary['by_status']:
                        summary['by_status'][status] = 0
                    summary['by_status'][status] += 1
            
            return summary
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return {}
