
#echo on
set -x

./rq3.sh 128 cuda:0 veneres/monobert-asnq /data/models/eemb/asnq/ /data/rq3/ asnq 1 ndcg
./rq3.sh 128 cuda:0 veneres/monobert-msmarco /data/models/eemb/msmarco/ /data/rq3/ msmarco-passage/dev 1 rr
./rq3.sh 128 cuda:0 veneres/monobert-msmarco /data/models/eemb/msmarco/ /data/rq3/ msmarco-passage/trec-dl-2019/judged 3 ndcg

