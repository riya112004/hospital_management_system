class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blockSize = 4096;
    this.buffer = new Float32Array(this.blockSize);
    this.offset = 0;
    this.sumSquares = 0;
    this.recycledBuffers = [];
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this.sendSamples();
        this.sumSquares = 0;
        this.port.postMessage({ type: "flushed" });
      } else if (event.data?.type === "recycle") {
        this.recycledBuffers.push(event.data.buffer);
      }
    };
  }

  sendSamples() {
    if (this.offset === 0) return;
    const buffer = this.buffer.buffer;
    const length = this.offset;
    this.port.postMessage({ type: "samples", buffer, length }, [buffer]);
    this.buffer = new Float32Array(
      this.recycledBuffers.pop() || new ArrayBuffer(this.blockSize * 4),
    );
    this.offset = 0;
  }

  process(inputs, outputs) {
    for (const output of outputs) {
      for (const channel of output) channel.fill(0);
    }

    const input = inputs[0]?.[0];
    if (!input) return true;

    for (const sample of input) {
      this.buffer[this.offset] = sample;
      this.sumSquares += sample * sample;
      this.offset += 1;
      if (this.offset === this.blockSize) {
        this.sendSamples();
        this.port.postMessage({
          type: "level",
          rms: Math.sqrt(this.sumSquares / this.blockSize),
        });
        this.sumSquares = 0;
      }
    }
    return true;
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
