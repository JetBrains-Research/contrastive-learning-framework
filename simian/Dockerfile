FROM openjdk:8

RUN wget https://www.harukizaemon.com/simian/simian-2.5.10.tar.gz
RUN mkdir simian
RUN tar -xvzf simian-2.5.10.tar.gz -C simian

COPY ./run_simian.sh .
RUN chmod +x run_simian.sh

CMD [ "/bin/bash" ]