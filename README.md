# wf-batch-practical

Edit .env

```
source .env
```

```
gcloud builds submit   -t us-central1-docker.pkg.dev/$PROJECT_ID/containers/featurize:v1 /home/admin_/batch-demo/wf-batch-practical/container/1.featurize/1.1.featurize-inParallel/
```

```
gcloud builds submit   -t us-central1-docker.pkg.dev/$PROJECT_ID/containers/feature-aggregation:v1 /home/admin_/batch-demo/wf-batch-practical/container/1.featurize/1.2.featuresAggregation/
```

```
gcloud builds submit   -t us-central1-docker.pkg.dev/$PROJECT_ID/containers/train:v1 /home/admin_/batch-demo/wf-batch-practical/container/2.train_and_eval/2.1.train-inParallel/
```

```
gcloud workflows deploy newsgroups-workflow   --source=/home/admin_/batch-demo/wf-batch-practical/workflows.yaml   --location=us-central1   --service-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com
```

```
gcloud workflows run newsgroups-workflow   --location=us-central1 >/dev/null
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
