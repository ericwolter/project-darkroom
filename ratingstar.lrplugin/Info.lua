local menuItems = {
     title = "Add automatic rating to selected photos...",
     file = "Init.lua",
     enabledWhen = "photosSelected"
}

return {
  LrSdkVersion = 6.0,
  LrToolkitIdentifier = "com.ericwolter.lightroom.ratingstar",
  LrPluginName = "EW/RatingStar",
  LrShutdownPlugin = "Shutdown.lua",

  LrExportMenuItems = menuItems,
  LrLibraryMenuItems = menuItems
}
