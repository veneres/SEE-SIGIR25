
#echo on
set -x


for lt in 100 500 1000
do
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/arguana 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/climate-fever 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/android 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/english 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/gaming 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/gis 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/mathematica 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/physics 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/programmers 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/stats 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/tex 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/unix 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/webmasters 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/cqadupstack/wordpress 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/dbpedia-entity/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/fever/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/fiqa/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/hotpotqa/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/msmarco/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/nfcorpus/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/nq 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/quora/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/scidocs 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/scifact/test 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/trec-covid 1 $lt
  ./rq4.sh 128 cuda:0 /data/models/eemb/msmarco/ /data/rq4/ beir/webis-touche2020/v2 1 $lt
done;

