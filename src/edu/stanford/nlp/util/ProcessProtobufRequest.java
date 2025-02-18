/**
 * An abstract class with a command line program for the processing of protobuf requests.
 *<br>
 * For example, the Semgrex version of this can process semgrexes over SemanticGraphs.
 * This will compile a given semgrex expression, build SemanticGraph objects,
 * and return the results of those objects
 */

package edu.stanford.nlp.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.InputStream;
import java.io.IOException;
import java.io.OutputStream;

public abstract class ProcessProtobufRequest {
  /**
   * Reads a single request from the InputStream, then writes back a single response.
   */
  public abstract void processInputStream(InputStream in, OutputStream out) throws IOException;

  /**
   * Processes multiple requests from the same stream.
   *<br>
   * As per the google suggestion for streaming multiple messages,
   * this reads the length of the buffer, then reads exactly that many
   * bytes and decodes it.  It repeats until either 0 is read for the
   * length or until EOF.
   *<br>
   * https://developers.google.com/protocol-buffers/docs/techniques#streamimg
   */
  public void processMultipleInputs(InputStream in, OutputStream out) throws IOException {
    DataInputStream din = new DataInputStream(in);
    DataOutputStream dout = new DataOutputStream(out);
    int size = 0;
    do {
      try {
        size = din.readInt();
      } catch (EOFException e) {
        // If the stream ends without a closing 0, we consider that okay too
        size = 0;
      }

      // stream is done if there's a closing 0 or if the stream ends
      if (size == 0) {
        dout.writeInt(0);
        break;
      }

      byte[] inputArray = new byte[size];
      din.read(inputArray, 0, size);
      ByteArrayInputStream bin = new ByteArrayInputStream(inputArray);
      ByteArrayOutputStream result = new ByteArrayOutputStream();
      processInputStream(bin, result);
      byte[] outputArray = result.toByteArray();
      dout.writeInt(outputArray.length);
      dout.write(outputArray);
    } while (size > 0);
  }

  public static void process(ProcessProtobufRequest processor, String[] args) throws IOException {
    if (args.length > 0 &&
        (args[0].equalsIgnoreCase("-multiple") || args[0].equalsIgnoreCase("--multiple"))) {
      processor.processMultipleInputs(System.in, System.out);
    } else {
      processor.processInputStream(System.in, System.out);
    }    
  }
}


