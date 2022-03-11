FROM adoptopenjdk/maven-openjdk13

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
                       build-essential \
                       wget \
                       curl \
                       git \
                       sudo \
                       vim


RUN git clone https://github.com/jplag/JPlag.git
WORKDIR /JPlag
RUN git checkout fbded8822d02c852fdc017e225a3d036fe417285
RUN sed '1432 s/./\/\/&/' jplag/src/main/java/jplag/Program.java > jplag/src/main/java/jplag/Program_tmp.java
RUN rm jplag/src/main/java/jplag/Program.java
RUN mv jplag/src/main/java/jplag/Program_tmp.java jplag/src/main/java/jplag/Program.java

RUN mvn clean generate-sources package
RUN mvn clean install

WORKDIR /JPlag/jplag
RUN mvn clean generate-sources assembly:assembly
RUN mv target/jplag-2.12.1-SNAPSHOT-jar-with-dependencies.jar ../..

WORKDIR /
COPY ./run_jplag.sh .
RUN chmod +x run_jplag.sh

CMD [ "/bin/sh", "run_jplag.sh" ]