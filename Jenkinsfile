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
                    docker exec lifemap_api pytest /app/api/test_suite.py
                '''
            }
        }
        stage('Build & Deploy Compose') {
            steps {
                withCredentials([string(credentialsId: 'kaggle-creds', variable: 'KAGGLE_API_TOKEN')]) {
                    sh '''
                        echo "Kaggle Secret Text token loaded securely into KAGGLE_API_TOKEN"
                        docker exec lifemap_api curl -sf http://localhost:5000/health
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
