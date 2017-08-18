local Require = require "Require".path ("../debuggingtoolkit.lrdevplugin").reload()
local Debug = require "Debug"
require "strict"

local LrLogger = import 'LrLogger'

local logging = {
  myLogger = LrLogger( 'RatingStar' )
}

logging.myLogger:enable( "logfile" ) -- Pass either a string or a table of actions.

function logging:log( message )
  self.myLogger:trace( message )
end

return logging
