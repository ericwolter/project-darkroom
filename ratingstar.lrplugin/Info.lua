local menuItems = {
     title = "Add automatic rating to selected photos...",
     file = "RatingStar.lua",
     enabledWhen = "photosSelected"
}

return {
  LrSdkVersion = 6.0,
  LrToolkitIdentifier = "com.ericwolter.lightroom.ratingstar",
  LrPluginName = "EW/RatingStar",

  LrExportMenuItems = menuItems,
  LrLibraryMenuItems = menuItems
}
