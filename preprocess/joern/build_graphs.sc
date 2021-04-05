/* graph-for-funcs.scala
   This script returns a Json representation of the graph
   Input:
    - cpgPath:    String  -- path to .bin file contains cpg
    - outputPath: String  -- where to store json
 */

import io.circe.Json
import io.circe.syntax._
import io.shiftleft.codepropertygraph.generated.nodes._

val skip_types =
  List("FILE", "UNKNOWN", "META_DATA", "NAMESPACE", "NAMESPACE_BLOCK")

val selectKeys =
  (v: StoredNode) => {
    if (v.propertyKeys.contains("NAME"))
      v.property("NAME").toString
    else if (v.propertyKeys.contains("FULL_NAME"))
      v.property("FULL_NAME").toString
    else if (v.propertyKeys.contains("TYPE_FULL_NAME"))
      v.property("TYPE_FULL_NAME").toString
    else
      v.property("CODE").toString
  }

@main def main(cpgPath: String, outputPath: String) = {
  importCpg(inputPath = cpgPath)

  val ids_map = cpg.all
    .filterNot(v => skip_types.contains(v.label))
    .map(v => v.id)
    .zipWithIndex
    .toMap

  val vertexes_json = cpg.all
    .filterNot(v => skip_types.contains(v.label))
    .map(v =>
      Map(
          "label" -> v.label,
          "id" -> ids_map(v.id),
          "name" -> selectKeys(v)
        )
    )
    .toJson
    .asJson

  val edges_json = cpg.graph.E
    .filter(e => ids_map.contains(e.inNode.id) & ids_map.contains(e.outNode.id))
    .map(e =>
      Map(
          "label" -> e.label,
          "in" -> ids_map(e.inNode.id),
          "out" -> ids_map(e.outNode.id)
        )
    )
    .toJson
    .asJson

  val output = Json
    .obj(
      ("vertexes", vertexes_json),
      ("edges", edges_json)
    )
    .toString

  output |> outputPath
}
