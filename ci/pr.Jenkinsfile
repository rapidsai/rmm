pipeline {
  agent any
  stages {
    stage("build") {
      steps {
        withRemoteDocker(debug: 'true',
            main: image(
                image: 'gpuci/rapidsai-driver:22.02-cuda11.5-devel-centos7-py3.8',
                forcePull: 'true',
            ),
            sideContainers: [],
            workspaceOverride: "",
            registryUrl: "",
            credentialsId: "",
        ) {
            sh """
              echo "hello world!"
            """
        }
      }
    }
  }
}

