// Move to shared library
test_configs  = [
  [label: "driver-450", cuda_ver: "11.0", py_ver: "3.8", os: "centos7"],
  [label: "driver-495", cuda_ver: "11.2", py_ver: "3.9", os: "ubuntu18.04"],
  [label: "driver-495", cuda_ver: "11.5", py_ver: "3.9", os: "ubuntu20.04"],
]

// Move to shared library
def generateStage(test_config, test_type) {
    return {
        stage("${test_type} Test - ${test_config.label} - ${test_config.cuda_ver} - ${test_config.py_ver} - ${test_config.os}") {
            node(test_config.label) {
                docker.image("gpuci/rapidsai:22.04-cuda${test_config.cuda_ver}-devel-${test_config.os}-py${test_config.py_ver}").inside("--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$EXECUTOR_NUMBER") {
                  sh "echo 'hello, tests'"
                  sh "nvidia-smi"
              }
            }
        }
    }
}

def getTestStagesMap(test_type) {
   return test_configs.collectEntries {
      ["${it.label} - ${it.cuda_ver} - ${it.py_ver} - ${it.os}" : generateStage(it, test_type)]
  }
}

pipeline {
  agent any
  stages {
    stage("C++ Build") {
      agent {
        docker {
            image 'gpuci/rapidsai:22.04-cuda11.5-devel-centos7-py3.8'
            label 'cpu4'
        }
      }
      steps {
        sh """
          #!/bin/bash
          # set +x
          # export CUDA=11.5
          # export PARALLEL_LEVEL=8
          # export CMAKE_GENERATOR=Ninja
          # env
          # conda info
          # conda env list
          # . /opt/conda/etc/profile.d/conda.sh
          # conda activate rapids
          # conda build --no-test --no-build-id conda/recipes/librmm
          echo "hello"
        """
      }
    }
    stage('C++ Tests & Python Build') {
      parallel {
        stage("C++ Package Tests") {
          agent {
            label 'cpu'
          }
          steps {
            script {
                parallel getTestStagesMap("C++")
            }
          }
        }
        stage("Python Build & Tests") {
          stages {
            stage("Python Build") {
              steps {
                sh "echo 'hello, python build'"
              }
            }
            stage("Python Package Tests") {
              steps {
                script {
                    parallel getTestStagesMap("Python")
                }
              }
            }
          }
        }
      }
    }
  }
}

