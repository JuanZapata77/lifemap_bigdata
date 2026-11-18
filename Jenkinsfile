pipeline {
    agent any
    triggers {
        pollSCM('H/2 * * * *')
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git rev-parse --short HEAD > commit.txt'
            }
        }
        stage('Unit & E2 Streaming Smoke Tests') {
            steps {
                sh 'docker exec lifemap_dask_scheduler python /app/jobs/kafka_producer.py --limit 100 --fast'
                sh 'docker exec lifemap_dask_scheduler python /app/jobs/stream_processor.py --batches 5'
                sh 'docker exec lifemap_api pytest /app/test_suite.py -v'
            }
        }
        stage('Deploy & Sync Commit') {
            steps {
                withCredentials([string(credentialsId: 'kaggle-key', variable: 'KAGGLE_KEY')]) {
                    sh 'docker cp commit.txt lifemap_api:/app/commit.txt'
                }
            }
        }
        stage('Smoke Test Live /health') {
            steps {
                sh 'curl -sf http://api:5000/health | grep -q "kafka"'
            }
        }
    }
}
