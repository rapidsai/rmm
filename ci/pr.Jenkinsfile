@Library('gpuci_shared_lib') _

pipeline {
  agent any
  stages {
    stage("C++ Matrix Build") {
      matrix {
        agent {
          docker {
            image 'gpuci/rapidsai:22.04-cuda11.5-devel-centos7-py3.8'
            label "cpu4-${ARC}"
          }
        }

        axes {
          axis {
            name 'ARC'
            values 'arm64', 'amd64'
          }
        }

        stages {
          stage ('C++ Build') {
            steps {
              echo "Do Build for ${ARC}"
            }
          }
        }      
      }
    }

    stage("Python Matrix Build") {
      matrix {
        agent {
          label 'cpu'
        }

        axes {
          axis {
            name 'ARC'
            values 'arm64', 'amd64'
          }

          axis {
            name 'PYTHON_VER'
            values '3.8', '3.9'
          }
        }

        stages {
          stage('Python Build') {
            steps {
              echo "Do Build for arc:${ARC} and py_ver:${PYTHON_VER}"
            }
          }
        }
      }
    }

    stage("Tests") {
      steps {
        script {
          parallel getTestStages({
            echo "hello from rmm repo"
          })
        }
      }
    }
  }
}