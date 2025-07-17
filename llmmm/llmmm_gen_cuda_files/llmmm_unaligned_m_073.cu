#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_73_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<73> mm_instantiator;

public:
  UnalignedM_73_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_73_MMInstantiatorWrapper__;

}  // namespace LLMMM
