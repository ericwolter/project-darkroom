local Require = require "Require".path ("../debuggingtoolkit.lrdevplugin").reload()
local Debug = require "Debug"
require "strict"

LOCALHOST_IP = "127.0.0.1"
PORT = 31414
CLIENT_ENDPOINT = "http://"..LOCALHOST_IP..":"..PORT.."/?path="
