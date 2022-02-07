pipeline {
  agent any
  stages {
    stage("build") {
      agent {
        docker {
            image 'gpuci/rapidsai-driver:22.02-cuda11.5-devel-centos7-py3.8'
            label 'cpu4'
        }
      }
      steps {
        sh """
          env
          echo "hello world!"
        """
      }
    }
  }
}

