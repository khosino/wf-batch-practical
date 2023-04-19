# wf-batch-practical

```
gcloud builds submit   -t us-central1-docker.pkg.dev/$PROJECT_ID/containers/featurize:v1 ./
```

```
gcloud beta batch jobs submit job-f10 --location us-central1 --config - <<EOD
{
  "name": "projects/cloud-batch-hclsj-demo/locations/us-central1/jobs/job-f10",
  "taskGroups": [
    {
      "taskCount": "2",
      "parallelism": "2",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "1000",
          "memoryMib": "512"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "us-central1-docker.pkg.dev/cloud-batch-hclsj-demo/containers/featurize:v1",
              "entrypoint": "",
              "volumes": []
            },
            "environment": {
                "variables": {
                    "BUCKET": "batch-test-keihoshino01",
                    "num_tasks_featurizer": "10"
                }
            }
          }
        ],
        "volumes": [
          {
            "gcs": {
              "remotePath": ""
            },
            "mountPath": ""
          }
        ]
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "provisioningModel": "STANDARD",
          "machineType": "e2-medium"
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOD
```


```
docker run  --env BUCKET=batch-test-keihoshino01 --env num_tasks_featurizer=10 us-central1-docker.pkg.dev/cloud-batch-hclsj-demo/containers/featurize:v1
```
