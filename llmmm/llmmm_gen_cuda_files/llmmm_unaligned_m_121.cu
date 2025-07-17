#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_121_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<121> mm_instantiator;

public:
  UnalignedM_121_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_121_MMInstantiatorWrapper__;

}  // namespace LLMMM
