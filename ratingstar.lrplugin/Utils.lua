local Require = require "Require".path ("../debuggingtoolkit.lrdevplugin").reload()
local Debug = require "Debug"
require "strict"

function urlencode(str)
   if (str) then
      str = string.gsub (str, "\n", "\r\n")
      str = string.gsub (str, "([^%w ])",
         function (c) return string.format ("%%%02X", string.byte(c)) end)
      str = string.gsub (str, " ", "+")
   end
   return str
end

function score2rating(score)
  local input_start = 0.3
  local input_end = 0.6
  local output_start = 1
  local output_end = 5
  local slope = 1.0 * (output_end - output_start) / (input_end - input_start)
  local rating = output_start + math.floor((slope * (score - input_start) + 0.5))
  rating = math.max(rating, output_start)
  rating = math.min(rating, output_end)

  return rating
end
