FROM adoptopenjdk/openjdk14:alpine-jre

RUN wget https://github.com/jplag/JPlag/releases/download/v3.0.0/jplag-3.0.0-jar-with-dependencies.jar

COPY ./prepare_test_set.sh .
RUN chmod +x prepare_test_set.sh

CMD [ "/bin/sh", "prepare_test_set.sh" ]