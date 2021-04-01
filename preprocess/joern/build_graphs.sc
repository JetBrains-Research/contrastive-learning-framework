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

  val valid_vert_ids = cpg.all
    .filterNot(v => skip_types.contains(v.label))
    .map(v => v.id)
    .toList

  val vertexes_json = cpg.all
    .filterNot(v => skip_types.contains(v.label))
    .map(v =>
      Json
        .obj(
          ("label", v.label.asJson),
          ("id", v.id.asJson),
          ("name", selectKeys(v).asJson)
        )
        .toString
    )
    .toJson
    .asJson

  val edges = cpg.graph.E
    .filter(e =>
      valid_vert_ids.contains(e.inNode.id) & valid_vert_ids.contains(
        e.outNode.id
      )
    )

  val edges_json = edges
    .map(e =>
      Json
        .obj(
          ("label", e.label.asJson),
          ("in", e.inNode.id.asJson),
          ("out", e.outNode.id.asJson)
        )
        .toString
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
