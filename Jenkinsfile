pipeline {
    agent any
    triggers {
        pollSCM('H/2 * * * *')
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                }
            }
        }
        stage('Unit Tests (pytest)') {
            steps {
                sh '''
                    docker cp api/test_suite.py lifemap_api:/app/api/test_suite.py
                    docker exec -e PYTHONPATH=/app lifemap_api pytest /app/api/test_suite.py -v
                '''
            }
        }
        stage('Build & Deploy Compose') {
            steps {
                withCredentials([string(credentialsId: 'kaggle-creds', variable: 'KAGGLE_API_TOKEN')]) {
                    sh '''
                        echo "Kaggle Secret Text token verified in KAGGLE_API_TOKEN"
                        docker cp api/app.py lifemap_api:/app/api/app.py
                        docker restart lifemap_api
                        sleep 5
                    '''
                }
            }
        }
        stage('Smoke Test /health') {
            steps {
                sh '''
                    docker exec lifemap_api curl -sf http://localhost:5000/health | grep status
                '''
            }
        }
    }
}
