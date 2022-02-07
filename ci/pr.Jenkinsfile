pipeline {
  agent any
  stages {
    stage("build") {
      agent {
        docker {
            image 'gpuci/rapidsai:22.02-cuda11.5-devel-centos7-py3.8'
            label 'driver-495'
            args "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$EXECUTOR_NUMBER"
        }
      }
      steps {
        sh """
          env
          echo "hello world!"
          nvidia-smi
        """
      }
    }
  }
}

