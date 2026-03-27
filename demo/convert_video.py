import ffmpeg

input_file = "/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/1736557631421.mp4"
output_file = "/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/1736557631421_converted.mp4"

(
    ffmpeg
    .input(input_file)
    .output(output_file, vcodec='libx264', preset='medium', crf=23)
    .run()
)