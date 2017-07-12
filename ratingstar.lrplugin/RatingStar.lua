local LrLogger = import 'LrLogger'
local LrApplication = import 'LrApplication'
local LrTasks = import 'LrTasks'
local LrPathUtils = import 'LrPathUtils'
local LrHttp = import 'LrHttp'

local myLogger = LrLogger( 'exportLogger' )
myLogger:enable( "logfile" )

local thumbnailDir = LrPathUtils.getStandardFilePath('temp')

function urlencode(str)
   if (str) then
      str = string.gsub (str, "\n", "\r\n")
      str = string.gsub (str, "([^%w ])",
         function (c) return string.format ("%%%02X", string.byte(c)) end)
      str = string.gsub (str, " ", "+")
   end
   return str
end

LrTasks.startAsyncTask( function()
  -- LrTasks.startAsyncTask( function()
  --   LrTasks.execute('python C:\\Users\\Home\\Documents\\GitHub\\project-darkroom\\ratingstar.py')
  -- end )

  local catalog = LrApplication.activeCatalog()

  local photos = catalog:getTargetPhotos()

  for _, photo in ipairs(photos) do
    myLogger:trace('Saving photo', photo)
    photo:requestJpegThumbnail(299, 299, function(data, err)
      if err == nil then
        local filepath = photo.path
        local filename = LrPathUtils.leafName(filepath)
        local path = LrPathUtils.child(thumbnailDir, filename)
        local jpgpath = LrPathUtils.addExtension(path, 'jpg')
        myLogger:trace('Got jpeg thumbnail typeof', type(data))
        myLogger:trace(data)
        myLogger:trace('Saving thumbnail to', jpgpath)
        local f = io.open(jpgpath, 'wb')
        f:write(data)
        f:close()

        local url = 'http://127.0.0.1:31414/?path=' .. urlencode(jpgpath)
        myLogger:trace('Request score', url)
        res, info = LrHttp.get(url, nil, 10)
        score = tonumber(res)
        input_start = 3
        input_end = 9
        output_start = 1
        output_end = 5
        slope = 1.0 * (output_end - output_start) / (input_end - input_start)
        rating = output_start + math.floor((slope * (score - input_start) + 0.5))
        rating = math.max(rating, output_start)
        rating = math.min(rating, output_end)

        myLogger:trace('Score:', res)

        catalog:withWriteAccessDo('Set rating', function(context)
          photo:setRawMetadata('rating', rating)
        end )
      else
        myLogger:error(err)
      end
    end )
  end

end )
