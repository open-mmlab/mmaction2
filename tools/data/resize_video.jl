using DelimitedFiles
using ArgParse
using MLDataPattern
using Distributed
addprocs()
@everywhere using Shell
@everywhere using ProgressMeter

s = ArgParseSettings()
@add_arg_table! s begin
    "--src", "-s"
        help = "source dir"
        arg_type = String
        default = "./data/kinetics400/videos"
    "--dst", "-t"
        help = "target dir"
        arg_type = String
        default = "./data/kinetics400/fast_256_videos"
    "--split"
        help = "which split to process"
        arg_type = Int
        default = 1
    "--splits"
        help = "How many splits"
        arg_type = Int
        default = 5
end
args = parse_args(s)

src = args["src"]
dst = args["dst"]
split = args["split"]
splits = args["splits"]
workdir = "./tmp"
rm(workdir, recursive=true, force=true)

mkpath(workdir)
mkpath(joinpath(workdir, "original"))
mkpath(joinpath(workdir, "resize"))

videos = splitobs(readdlm("kinetics400_train_list_videos.txt")[:, 1], at=tuple(ones(splits-1)/splits...))[split]
println("processing split $split/$splits, $(length(videos)) videos")

errors = @showprogress pmap(videos) do f
    try
        # resize
        (w, h) = eval(Meta.parse(Shell.run("""
        ffprobe -hide_banner -loglevel panic -select_streams v:0 -show_entries stream=width,height -of csv=p=0 '.$src/$f'
        """, capture=true)))
        if w <= 256 || h <= 256
            Shell.run("""
            ffmpeg -hide_banner -loglevel panic -i '$src/$f' -vf mpdecimate -vsync vfr -c:v libx264 -g 16 -an '$dst/$f' -y
            """)
        elseif w > h
            Shell.run("""
            ffmpeg -hide_banner -loglevel panic -i '$src/$f' -vf mpdecimate,scale=-2:256 -vsync vfr -c:v libx264 -g 16 -an '$dst/$f' -y
            """)
        else
            Shell.run("""
            ffmpeg -hide_banner -loglevel panic -i '$src/$f' -vf mpdecimate,scale=256:-2 -vsync vfr -c:v libx264 -g 16 -an '$dst/$f' -y
            """)
        end

    catch
        println("error: ", f)
        return f
    end
    return nothing
end

errors = filter(!isnothing, errors)
writedlm("resize-split$(split)_of_$(splits)-errors.log", errors)
