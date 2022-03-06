test_configs  = [
  [label: "driver-450", cuda_ver: "11.0", py_ver: "3.8", os: "centos7", arc: "ARM"],
  [label: "driver-495", cuda_ver: "11.2", py_ver: "3.9", os: "ubuntu18.04", arc: "ARM"],
  [label: "driver-495", cuda_ver: "11.5", py_ver: "3.9", os: "ubuntu20.04", arc: "ARM"],
  
  // repeat the same combinations for AMD arc
  [label: "driver-450", cuda_ver: "11.0", py_ver: "3.8", os: "centos7", arc: "AMD"],
  [label: "driver-495", cuda_ver: "11.2", py_ver: "3.9", os: "ubuntu18.04", arc: "AMD"],
  [label: "driver-495", cuda_ver: "11.5", py_ver: "3.9", os: "ubuntu20.04", arc: "AMD"],
]

def generateStage(test_config) {
    return {
        stage("Test - ${test_config.label} - ${test_config.cuda_ver} - ${test_config.py_ver} - ${test_config.os}") {
          node(test_config.label) {
              docker.image("gpuci/rapidsai:22.04-cuda${test_config.cuda_ver}-devel-${test_config.os}-py${test_config.py_ver}").inside("--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$EXECUTOR_NUMBER") {
                sh "echo 'hello, tests'"
                sh "nvidia-smi"
            }
          }
        }
    }
}

def getTestStagesMap() {
   return test_configs.collectEntries {
      ["${it.arc}: ${it.label} - ${it.cuda_ver} - ${it.py_ver} - ${it.os}" : generateStage(it)]
  }
}

pipeline {
  agent any
  stages {
    stage("C++ Matrix Build") {
      matrix {
        agent {
          docker {
            image 'gpuci/rapidsai:22.04-cuda11.5-devel-centos7-py3.8'
            label 'cpu4'
          }
        }

        axes {
          axis {
            name 'ARC'
            values 'ARM', 'AMD'
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
            values 'ARM', 'AMD'
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
            parallel getTestStagesMap()
        }
      }
    }
  }
}