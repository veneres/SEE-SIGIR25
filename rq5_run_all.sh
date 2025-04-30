
#echo on
set -x



./rq5.sh 128 cuda:0 /data/rq5/ beir/arguana 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/climate-fever 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/android 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/english 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/gaming 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/gis 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/mathematica 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/physics 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/programmers 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/stats 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/tex 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/unix 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/webmasters 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/cqadupstack/wordpress 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/dbpedia-entity/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/fever/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/fiqa/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/hotpotqa/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/msmarco/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/nfcorpus/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/nq 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/quora/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/scidocs 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/scifact/test 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/trec-covid 1
./rq5.sh 128 cuda:0 /data/rq5/ beir/webis-touche2020/v2 1

