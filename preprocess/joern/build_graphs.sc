/* graph-for-funcs.scala
   This script returns a Json representation of the graph
   Input:
    - cpgPath:    String  -- path to .bin file contains cpg
    - outputPath: String  -- where to store json
 */

import io.circe._
import io.circe.generic.auto._
import io.circe.parser._
import io.circe.syntax._
import io.circe.generic.semiauto._
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.nodes._
import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

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

def build_graph(cpgPath: String, outputPath: String) = {
  val Some(cpg) = importCpg(cpgPath)

  val valid_v = cpg.all.filterNot(v => skip_types.contains(v.label)).toList
  val ids_map = valid_v
    .map(v => v.id)
    .zipWithIndex
    .toMap

  val vertexes = Future.traverse(valid_v) { v =>
    Future {
      Map(
        "label" -> v.label,
        "id" -> ids_map(v.id).toString,
        "name" -> selectKeys(v)
      )
    }
  }

  val vertexes_json = Await.result(vertexes, Duration.Inf).asJson

  val valid_e = cpg.graph.E
    .filter(e => ids_map.contains(e.inNode.id) & ids_map.contains(e.outNode.id))
    .toList
  val edges = Future.traverse(valid_e) { e =>
    Future {
      Map(
        "label" -> e.label,
        "in" -> ids_map(e.inNode.id).toString,
        "out" -> ids_map(e.outNode.id).toString
      )
    }
  }

  val edges_json = Await.result(edges, Duration.Inf).asJson

  val output = Json
    .obj(
      ("vertexes", vertexes_json),
      ("edges", edges_json)
    )
    .toString

  output |> outputPath
  close(workspace.projectByCpg(cpg).map(_.name).get)
}

@main def main(inputPath: String, cpgPath: String, outputPath: String) = {
  val output_ = better.files.File(outputPath)
  val cpg_storage_ = better.files.File(cpgPath)

  val fileList = better.files
    .File(inputPath)
    .listRecursively
    .filter { e => e.isRegularFile }
    .filterNot { f =>
      (output_ / f.parent.name / (f.nameWithoutExtension + ".json")).exists
    }
    .toList

  val result = Future {
    fileList.foreach { f =>
      val cpg_path =
        cpg_storage_ / f.parent.name / (f.nameWithoutExtension + ".bin")
      val output_path =
        output_ / f.parent.name / (f.nameWithoutExtension + ".json")
      build_graph(cpg_path.pathAsString, output_path.pathAsString)
    }
  }
  Await.result(result, Duration.Inf)
}

