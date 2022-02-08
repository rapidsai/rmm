pipeline {
  agent any
  stages {
    stage("build") {
      agent {
        docker {
            image 'gpuci/rapidsai:22.02-cuda11.5-devel-centos7-py3.8'
            label 'cpu4'
        }
      }
      steps {
        sh """
          #!/bin/bash
          set +x
          export CUDA=11.5
          env
          conda info
          conda env list
          . /opt/conda/etc/profile.d/conda.sh
          conda activate rapids
          conda build --no-build-id --croot $WORKSPACE/.conda-bld conda/recipes/librmm
        """
      }
    }
  }
}

