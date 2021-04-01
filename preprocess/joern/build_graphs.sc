/* build_graphs.sc
   This script created a Json representation of the graph and saves it to outputPath
   Input:
    - cpgPath:    String  -- path to .bin file contains cpg
    - outputPath: String  -- where to store json
 */

import io.circe.{Encoder, Json}

@main def main(cpgPath: String, outputPath: String) = {
  importCpg(inputPath=cpgPath)

  val vertexes = cpg.all

  val edges = cpg.graph.E.map(
    e => Json.obj(
      ("label", Json.fromString(e.label)),
      ("in", Json.fromString(e.inNode.id.toString)),
      ("out", Json.fromString(e.outNode.id.toString))
    ).toString
  )

  val output = Json.obj(
    ("vertexes", Json.fromString(vertexes.toJson)),
    ("edges", Json.fromString(edges.toJson))
  ).toString

  output |> outputPath
}