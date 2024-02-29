package com.example.mlapp1

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val inputArea = findViewById<EditText>(R.id.input_area)
        val inputRooms = findViewById<EditText>(R.id.input_room)
        val predictBtn = findViewById<Button>(R.id.predict_button)
        val resultText = findViewById<TextView>(R.id.output_text)

        predictBtn.setOnClickListener{
            val areaData = inputArea.text.toString().toFloatOrNull()
            val roomData = inputRooms.text.toString().toFloatOrNull()

            if ( areaData != null && roomData != null){
                val ortEnvironment  = OrtEnvironment.getEnvironment()
                val ortSession = createORTSession( ortEnvironment )
                val output =
                    ortSession?.let { it1 -> executeModel(areaData, roomData, it1, ortEnvironment) }
                resultText.text = "Predicted price is ${output}"
            }else{
                Toast.makeText(this, "Please input the area and room data", Toast.LENGTH_LONG).show()
            }
        }

    }

    private fun executeModel(areaData: Float, roomData: Float, ortSession: OrtSession, ortEnvironment: OrtEnvironment?): Float {
        val getName = ortSession.inputNames?.iterator()?.next()
        val floatBufferInput = FloatBuffer.wrap(floatArrayOf(areaData, roomData))
        val tensorInput = OnnxTensor.createTensor( ortEnvironment, floatBufferInput, longArrayOf(1,2))
        val result = ortSession.run(mapOf(getName to tensorInput))
        val output = result[0].value as Array<FloatArray>
        return output[0][0]
    }

    private fun createORTSession(ortEnvironment: OrtEnvironment?): OrtSession? {
        val modelFile = resources.openRawResource(R.raw.house_price_model).readBytes()

        return ortEnvironment?.createSession(modelFile)


    }
}