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
        stage('Run Unit Tests') {
            steps {
                sh 'docker run --rm -v "$PWD":/app -w /app python:3.11-slim bash -c "pip install -q -r api/requirements.txt && pytest tests/test_lifemap.py"'
            }
        }
        stage('Build & Deploy Compose') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'kaggle-creds', usernameVariable: 'KAGGLE_USERNAME', passwordVariable: 'KAGGLE_KEY')]) {
                    sh '''
                        export GIT_COMMIT=${GIT_COMMIT}
                        docker compose up -d --build
                    '''
                }
            }
        }
        stage('Smoke Test /health') {
            steps {
                sh '''
                    sleep 10
                    curl -f http://localhost:5000/health | grep ${GIT_COMMIT}
                '''
            }
        }
    }
}